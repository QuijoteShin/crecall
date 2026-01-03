# src/core/memory_manager.py
"""Context-aware GPU memory manager for pipeline optimization."""

import gc
from typing import Dict, Any, Optional, List, Protocol
from dataclasses import dataclass, field

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUComponent(Protocol):
    """Protocol for components that use GPU memory."""
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def get_memory_usage_mb(self) -> float: ...


@dataclass
class ComponentState:
    """Tracks runtime state of a GPU component."""
    component: Any  # GPUComponent (Any to avoid Protocol issues)
    name: str
    estimated_memory_mb: float
    component_type: str  # 'classifier', 'refiner', 'vectorizer'
    is_loaded: bool = False
    priority: int = 0  # Higher = keep loaded longer


class MemoryManager:
    """
    Context-aware GPU memory manager.

    Optimizations:
    1. If Stage N and N+1 use same model → NO unload between
    2. If both models fit in VRAM → keep both loaded
    3. Unload ONLY when: next stage needs model that won't fit, or pipeline done

    Design principles:
    - Avoid thrashing (load/unload cycles)
    - Maximize GPU utilization within limits
    - Predictable behavior via explicit prepare_for_stage() calls
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.total_vram_mb = self._detect_total_vram()
        self.safety_margin_mb = float(self.config.get("safety_margin_mb", 500))
        self.available_vram_mb = self.total_vram_mb - self.safety_margin_mb
        self.context_aware = self.config.get("context_aware", True)

        self._components: Dict[str, ComponentState] = {}
        self._load_order: List[str] = []  # Track order for LIFO unload

    def _detect_total_vram(self) -> float:
        """Detect available GPU memory."""
        if not TORCH_AVAILABLE:
            return 0.0

        if not torch.cuda.is_available():
            return 0.0

        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / 1024 / 1024
        except Exception:
            return 0.0

    def register_component(
        self,
        name: str,
        component: Any,
        estimated_memory_mb: float,
        component_type: str,
        priority: int = 0
    ) -> None:
        """
        Register a GPU component for memory management.

        Args:
            name: Unique identifier (e.g., 'classifier', 'refiner')
            component: Object implementing load()/unload()/get_memory_usage_mb()
            estimated_memory_mb: Expected VRAM usage when loaded
            component_type: Category for grouping
            priority: Higher values = keep loaded longer when freeing memory
        """
        self._components[name] = ComponentState(
            component=component,
            name=name,
            estimated_memory_mb=estimated_memory_mb,
            component_type=component_type,
            is_loaded=False,
            priority=priority
        )

    def get_current_usage_mb(self) -> float:
        """Get sum of estimated memory from loaded components."""
        return sum(
            state.estimated_memory_mb
            for state in self._components.values()
            if state.is_loaded
        )

    def get_actual_usage_mb(self) -> float:
        """Get actual GPU memory usage from CUDA."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1024 / 1024

    def can_fit(self, additional_mb: float) -> bool:
        """Check if additional memory can fit in available VRAM."""
        return (self.get_current_usage_mb() + additional_mb) <= self.available_vram_mb

    def is_loaded(self, name: str) -> bool:
        """Check if a component is currently loaded."""
        if name not in self._components:
            return False
        return self._components[name].is_loaded

    def prepare_for_stage(
        self,
        required: List[str],
        next_stage: Optional[List[str]] = None
    ) -> None:
        """
        Prepare memory for a pipeline stage.

        Smart decisions:
        - Load required components if not already loaded
        - Keep components loaded if needed by next stage
        - Unload only if necessary to fit new components

        Args:
            required: Component names needed for current stage
            next_stage: Component names needed for next stage (for look-ahead)
        """
        if not self.context_aware:
            # Fallback: simple load-process-unload
            self._simple_prepare(required)
            return

        # Calculate memory needed for required components not yet loaded
        memory_needed = sum(
            self._components[name].estimated_memory_mb
            for name in required
            if name in self._components and not self._components[name].is_loaded
        )

        # Check if we need to free memory
        if not self.can_fit(memory_needed):
            self._free_memory_for(memory_needed, keep=required, also_keep=next_stage)

        # Load required components
        for name in required:
            self._ensure_loaded(name)

    def _simple_prepare(self, required: List[str]) -> None:
        """Simple mode: just ensure required components are loaded."""
        for name in required:
            self._ensure_loaded(name)

    def _ensure_loaded(self, name: str) -> None:
        """Ensure a component is loaded."""
        if name not in self._components:
            return

        state = self._components[name]
        if state.is_loaded:
            return

        print(f"[MemoryManager] Loading: {name} (~{state.estimated_memory_mb:.0f} MB)")
        state.component.load()
        state.is_loaded = True

        # Track load order for LIFO unload
        if name in self._load_order:
            self._load_order.remove(name)
        self._load_order.append(name)

    def _free_memory_for(
        self,
        needed_mb: float,
        keep: List[str],
        also_keep: Optional[List[str]] = None
    ) -> None:
        """
        Free memory by unloading components not in keep lists.

        Uses priority + LIFO for unload order.
        """
        keep_set = set(keep)
        if also_keep:
            keep_set.update(also_keep)

        # Get candidates for unloading (loaded, not in keep set)
        candidates = [
            name for name in self._load_order
            if name not in keep_set and self._components[name].is_loaded
        ]

        # Sort by priority (lower first) then by load order (LIFO)
        candidates.sort(key=lambda n: (self._components[n].priority, -self._load_order.index(n)))

        # Unload until we have enough space
        for name in candidates:
            self._unload_component(name)

            if self.can_fit(needed_mb):
                break

    def _unload_component(self, name: str) -> None:
        """Unload a specific component."""
        if name not in self._components:
            return

        state = self._components[name]
        if not state.is_loaded:
            return

        print(f"[MemoryManager] Unloading: {name}")
        state.component.unload()
        state.is_loaded = False

        if name in self._load_order:
            self._load_order.remove(name)

    def finish_stage(self, completed: List[str], next_stage: Optional[List[str]] = None) -> None:
        """
        Called after a stage completes.

        In context-aware mode, decides whether to unload based on next stage needs.
        """
        if not self.context_aware:
            # Simple mode: unload everything from this stage
            for name in completed:
                self._unload_component(name)
            self._cleanup_gpu()
            return

        # Context-aware: only unload if not needed by next stage
        if next_stage:
            next_set = set(next_stage)
            for name in completed:
                if name not in next_set:
                    self._unload_component(name)
        # If no next stage, keep everything loaded (unload_all will be called later)

    def unload_all(self) -> None:
        """Unload all components (end of pipeline)."""
        print("[MemoryManager] Unloading all components...")

        # Unload in reverse order (LIFO)
        for name in reversed(self._load_order.copy()):
            self._unload_component(name)

        self._cleanup_gpu()
        print("[MemoryManager] All components unloaded")

    def _cleanup_gpu(self) -> None:
        """Force GPU memory cleanup."""
        gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_status(self) -> Dict[str, Any]:
        """Get current memory manager status for debugging."""
        return {
            "total_vram_mb": self.total_vram_mb,
            "available_vram_mb": self.available_vram_mb,
            "safety_margin_mb": self.safety_margin_mb,
            "estimated_usage_mb": self.get_current_usage_mb(),
            "actual_usage_mb": self.get_actual_usage_mb(),
            "context_aware": self.context_aware,
            "loaded_components": [
                {
                    "name": name,
                    "type": state.component_type,
                    "memory_mb": state.estimated_memory_mb,
                    "priority": state.priority
                }
                for name, state in self._components.items()
                if state.is_loaded
            ],
            "registered_components": list(self._components.keys()),
            "load_order": self._load_order.copy()
        }

    def log_status(self) -> None:
        """Print current status to console."""
        status = self.get_status()
        print(f"\n[MemoryManager Status]")
        print(f"  VRAM: {status['estimated_usage_mb']:.0f} / {status['available_vram_mb']:.0f} MB")
        print(f"  Loaded: {[c['name'] for c in status['loaded_components']]}")
