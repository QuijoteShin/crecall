# src/core/registry.py
"""Component registry for pluggable architecture."""

from typing import Dict, Type, Any, Optional
import importlib
import inspect


class ComponentRegistry:
    """
    Registry for dynamically loading components from configuration.

    Supports loading classes by their full Python path.
    Example: 'drecall.importers.chatgpt.ChatGPTImporter'
    """

    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._classes: Dict[str, Type] = {}

    def register_class(self, name: str, cls: Type) -> None:
        """
        Register a class manually.

        Args:
            name: Component name (e.g., 'chatgpt_importer')
            cls: Class to register
        """
        self._classes[name] = cls

    def load_class(self, class_path: str) -> Type:
        """
        Load a class from its full Python path.

        Args:
            class_path: Full path like 'drecall.importers.chatgpt.ChatGPTImporter'

        Returns:
            Class object

        Raises:
            ImportError: If module/class cannot be loaded
        """
        if class_path in self._classes:
            return self._classes[class_path]

        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            self._classes[class_path] = cls
            return cls

        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Cannot load class '{class_path}': {e}")

    def create_instance(
        self,
        name: str,
        class_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create and cache a component instance.

        Args:
            name: Instance name for caching
            class_path: Full class path
            config: Configuration dict to pass to constructor

        Returns:
            Component instance
        """
        if name in self._instances:
            return self._instances[name]

        cls = self.load_class(class_path)

        if config is None:
            config = {}

        # Inspect constructor to see if it accepts **kwargs
        sig = inspect.signature(cls.__init__)
        if 'config' in sig.parameters:
            instance = cls(config=config)
        elif len(sig.parameters) > 1:  # Has parameters beyond 'self'
            instance = cls(**config)
        else:
            instance = cls()

        self._instances[name] = instance
        return instance

    def get_instance(self, name: str) -> Optional[Any]:
        """
        Get cached instance by name.

        Returns None if not found.
        """
        return self._instances.get(name)

    def clear_instances(self) -> None:
        """Clear all cached instances."""
        self._instances.clear()

    def list_instances(self) -> Dict[str, Any]:
        """Return all cached instances."""
        return self._instances.copy()
