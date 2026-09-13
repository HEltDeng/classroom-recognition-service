from __future__ import annotations

import importlib.util
import os
from typing import Any

import numpy as np

from src.providers.base import TextRecognitionProvider, TextRecognitionResult
from src.schemas.models import RecognitionStatus


class DeepSeekOCR2BenchmarkProvider(TextRecognitionProvider):
    """Conditional DeepSeek-OCR2 candidate; not a production default."""

    name = "deepseek_ocr2_benchmark"

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR-2"):
        self.model_version = model_name
        self.min_vram_gb = float(os.getenv("DEEPSEEK_OCR2_MIN_VRAM_GB", "10"))
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_error: str | None = None

    def _cuda_is_usable(self) -> bool:
        if not importlib.util.find_spec("torch"):
            return False
        import torch

        if not torch.cuda.is_available():
            return False
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return memory_gb >= self.min_vram_gb

    @property
    def is_available(self) -> bool:
        return bool(
            importlib.util.find_spec("torch")
            and importlib.util.find_spec("transformers")
            and self._cuda_is_usable()
        )

    def _load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        if not self.is_available:
            self._load_error = (
                f"DeepSeek-OCR2 requires CUDA and at least {self.min_vram_gb:.1f} GB VRAM"
            )
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_version, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.model_version,
                trust_remote_code=True,
                use_safetensors=True,
            ).eval().cuda().to(torch.bfloat16)
        except Exception as exc:  # noqa: BLE001 - benchmark provider boundary
            self._load_error = f"{type(exc).__name__}: {exc}"

    def recognize_crop(self, crop: np.ndarray) -> TextRecognitionResult:
        # The full model expects an image file in its reference inference code;
        # the temporary file, when explicitly enabled, contains only this crop.
        self._load()
        if self._model is None or self._tokenizer is None:
            return TextRecognitionResult(
                raw_text=None,
                confidence=0.0,
                engine=self.name,
                status=RecognitionStatus.REVIEW_REQUIRED,
                metadata={"reason": self._load_error or "provider_unavailable"},
            )
        return TextRecognitionResult(
            raw_text=None,
            confidence=0.0,
            engine=self.name,
            status=RecognitionStatus.REVIEW_REQUIRED,
            metadata={"reason": "adapter_pending_reference_inference_integration"},
        )
