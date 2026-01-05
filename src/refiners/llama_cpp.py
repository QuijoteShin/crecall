# src/refiners/llama_cpp.py
"""Semantic refiner using llama-cpp-python with GGUF models."""

import gc
import re
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..contracts.refiner import IRefiner, RefinedOutput


class LlamaCppRefiner(IRefiner):
    """
    Semantic refiner using llama-cpp-python with quantized GGUF models.

    Default model: Qwen2.5-3B-Instruct-Q4_K_M.gguf
    - VRAM: ~2.2GB (Q4_K_M quantization)
    - Context: 2048 tokens
    - Languages: Excellent Spanish support

    Output format:
    "Intención: [intent] | Tema: [entities] | Resumen: [summary]"
    """

    DEFAULT_PROMPT_TEMPLATE = """<|im_start|>system
Eres un analizador semántico para búsqueda vectorial. Tu trabajo es EXPANDIR y ENRIQUECER el texto para mejorar la recuperación semántica.

REGLAS CRÍTICAS:
1. EXPANDE siglas y acrónimos según el CONTEXTO del mensaje:
   - SaaS en contexto tech → Software as a Service / Software como Servicio
   - API en programación → Application Programming Interface
   - API en farmacia → Active Pharmaceutical Ingredient
   - CRM → Customer Relationship Management / Gestión de Clientes
   - Usa el contexto para desambiguar siglas multi-significado
2. Incluye sinónimos y términos relacionados del dominio
3. El resumen debe ser DENSO semánticamente, optimizado para embeddings
4. DETECTA si hay pregunta o intención de pregunta:
   - Pregunta explícita: termina con "?" o usa palabras interrogativas
   - Pregunta implícita: solicita información, ayuda, o clarificación
   - Una pregunta indica cambio de contexto o nuevo tema jerárquico
<|im_end|>
<|im_start|>user
Analiza este mensaje y extrae:
1. Intención (máximo 3 palabras o 2 frases cortas)
2. Términos clave EXPANDIDOS: siglas → forma completa según contexto + sinónimos
3. Tiene_Pregunta: SI/NO (detecta preguntas explícitas o implícitas)
4. Resumen semántico denso (para búsqueda vectorial, no literal)

Mensaje:
{text}

Formato EXACTO:
Intención: [intención]
Términos: [término1 (expansión contextual), término2, ...]
Tiene_Pregunta: [SI/NO]
Resumen: [resumen denso con términos expandidos]
<|im_end|>
<|im_start|>assistant
"""

    # VRAM estimates for common quantizations
    VRAM_ESTIMATES = {
        "q4_k_m": 2200,
        "q5_k_m": 2800,
        "q8_0": 3500,
        "q4_0": 2000,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppRefiner. "
                "Install with: pip install llama-cpp-python"
            )

        self.config = config or {}
        self.model_path = Path(self.config.get(
            "model_path",
            "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
        ))
        self.n_gpu_layers = self.config.get("n_gpu_layers", -1)  # -1 = all to GPU
        self.n_ctx = self.config.get("n_ctx", 2048)
        self.max_tokens = self.config.get("max_tokens", 800)
        self.temperature = self.config.get("temperature", 0.1)  # Low for consistency
        self.prompt_template = self.config.get("prompt_template", self.DEFAULT_PROMPT_TEMPLATE)
        self.max_input_chars = self.config.get("max_input_chars", 1500)

        self.model: Optional[Llama] = None
        self._memory_mb = 0.0
        self._quantization = self._detect_quantization()

    def _detect_quantization(self) -> str:
        """Detect quantization from model filename."""
        name = self.model_path.name.lower()
        for quant in self.VRAM_ESTIMATES.keys():
            if quant.replace("_", "") in name.replace("-", "").replace("_", ""):
                return quant
        return "q4_k_m"  # Default assumption

    def refine(self, text: str) -> RefinedOutput:
        """Refine single text."""
        results = self.refine_batch([text])
        return results[0]

    def refine_batch(self, texts: List[str]) -> List[RefinedOutput]:
        """Batch refinement - processes sequentially (SLM limitation)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for text in texts:
            result = self._refine_single(text)
            results.append(result)

        return results

    def _refine_single(self, text: str) -> RefinedOutput:
        """Process single text through SLM."""
        # Handle empty/whitespace
        if not text or not text.strip():
            return RefinedOutput(
                refined_content="",
                confidence=0.0,
                metadata={"skipped": "empty_input"}
            )

        # Truncate long texts
        truncated = text[:self.max_input_chars]
        if len(text) > self.max_input_chars:
            truncated = truncated.rsplit(" ", 1)[0] + "..."

        # Build prompt
        prompt = self.prompt_template.format(text=truncated)

        try:
            output = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["<|im_end|>", "<|im_start|>", "\n\n\n"],
                echo=False
            )

            response = output["choices"][0]["text"].strip()
            return self._parse_response(response, text)

        except Exception as e:
            # Fallback: use truncated original text
            return RefinedOutput(
                refined_content=truncated[:500],
                confidence=0.0,
                metadata={"error": str(e), "fallback": True}
            )

    def _parse_response(self, response: str, original: str) -> RefinedOutput:
        """Parse structured SLM response into RefinedOutput."""
        intent = None
        entities = []
        summary = None
        has_question = False
        in_terms_block = False

        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                in_terms_block = False
                continue

            # Parse multi-line terms block (lines starting with -)
            if in_terms_block and line.startswith("-"):
                term = line.lstrip("- ").strip()
                if term:
                    entities.append(term)
                continue
            elif in_terms_block and not line.startswith("-"):
                in_terms_block = False

            # Parse each field
            if line.lower().startswith("intención:") or line.lower().startswith("intencion:"):
                intent = re.sub(r"^intenci[oó]n:\s*", "", line, flags=re.IGNORECASE).strip()
                intent = intent.strip("[]")
            elif line.lower().startswith("términos:") or line.lower().startswith("terminos:") or line.lower().startswith("entidades:"):
                ent_str = re.sub(r"^(t[eé]rminos|entidades):\s*", "", line, flags=re.IGNORECASE).strip()
                ent_str = ent_str.strip("[]")
                if ent_str:
                    entities = [e.strip() for e in ent_str.split(",") if e.strip()]
                else:
                    in_terms_block = True
            elif line.lower().startswith("tiene_pregunta:"):
                val = re.sub(r"^tiene_pregunta:\s*", "", line, flags=re.IGNORECASE).strip().upper()
                has_question = val in ["SI", "SÍ", "YES", "TRUE", "1"]
            elif line.lower().startswith("resumen:"):
                summary = re.sub(r"^resumen:\s*", "", line, flags=re.IGNORECASE).strip()

        # Build refined_content for vectorization
        parts = []
        if intent:
            parts.append(f"Intención: {intent}")
        if entities:
            parts.append(f"Términos: {', '.join(entities[:8])}")
        if summary:
            parts.append(f"Resumen: {summary}")

        # Fallback if parsing failed
        if not parts:
            refined_content = original[:500] if len(original) > 500 else original
            confidence = 0.3
        else:
            refined_content = " | ".join(parts)
            confidence = 0.9

        return RefinedOutput(
            refined_content=refined_content,
            intent=intent,
            entities=entities,
            summary=summary,
            confidence=confidence,
            metadata={
                "raw_response": response,
                "has_question": has_question
            }
        )

    def load(self) -> None:
        """Load GGUF model into GPU/CPU."""
        if self.model is not None:
            return  # Already loaded (idempotent)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {self.model_path}\n"
                f"Download from: https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF"
            )

        print(f"Loading Refiner: {self.model_path.name}...")
        print(f"  GPU layers: {self.n_gpu_layers}, Context: {self.n_ctx}")

        self.model = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False
        )

        # Estimate VRAM usage
        self._memory_mb = float(self.VRAM_ESTIMATES.get(self._quantization, 2200))

        print(f"✓ Refiner loaded (~{self._memory_mb:.0f} MB VRAM)")

    def unload(self) -> None:
        """Free model resources."""
        if self.model is None:
            return  # Already unloaded (idempotent)

        print("Unloading Refiner...")

        del self.model
        self.model = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._memory_mb = 0.0
        print("✓ Refiner unloaded")

    def get_memory_usage_mb(self) -> float:
        """Report current VRAM usage."""
        if self.model is None:
            return 0.0
        return self._memory_mb

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_path.stem,
            "quantization": self._quantization,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "max_tokens": self.max_tokens,
            "estimated_vram_mb": self.VRAM_ESTIMATES.get(self._quantization, 2200),
            "is_loaded": self.model is not None
        }
