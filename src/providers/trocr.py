from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from src.providers.base import TextRecognitionProvider, TextRecognitionResult
from src.schemas.models import RecognitionStatus


class TrOCRBenchmarkProvider(TextRecognitionProvider):
    """Optional benchmark-only handwritten OCR candidate.

    It is intentionally never selected by the service default. The model/data
    licensing review for commercial use is a separate gate.
    """

    name = "trocr_benchmark"

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.model_version = model_name
        self._processor: Any | None = None
        self._model: Any | None = None
        self._load_error: str | None = None

    @property
    def is_available(self) -> bool:
        return bool(
            importlib.util.find_spec("torch")
            and importlib.util.find_spec("transformers")
            and importlib.util.find_spec("PIL")
        )

    def _load(self) -> None:
        if self._processor is not None or self._load_error is not None:
            return
        if not self.is_available:
            self._load_error = "torch/transformers/Pillow are not installed"
            return
        try:
            from PIL import Image
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            del Image
            self._processor = TrOCRProcessor.from_pretrained(self.model_version)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_version)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 - benchmark provider boundary
            self._load_error = f"{type(exc).__name__}: {exc}"

    def recognize_crop(self, crop: np.ndarray) -> TextRecognitionResult:
        self._load()
        if self._model is None or self._processor is None:
            return TextRecognitionResult(
                raw_text=None,
                confidence=0.0,
                engine=self.name,
                status=RecognitionStatus.REVIEW_REQUIRED,
                metadata={"reason": self._load_error or "provider_unavailable"},
            )
        try:
            import torch
            from PIL import Image

            image = Image.fromarray(crop)
            inputs = self._processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated = self._model.generate(inputs, output_scores=True, return_dict_in_generate=True)
            text = self._processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
            return TextRecognitionResult(
                raw_text=text or None,
                confidence=0.5 if text else 0.0,
                engine=self.name,
                status=RecognitionStatus.RECOGNIZED if text else RecognitionStatus.REVIEW_REQUIRED,
                metadata={"license_gate": "PENDING_COMMERCIAL_MODEL_DATA_REVIEW"},
            )
        except Exception as exc:  # noqa: BLE001 - benchmark provider boundary
            return TextRecognitionResult(
                raw_text=None,
                confidence=0.0,
                engine=self.name,
                status=RecognitionStatus.REVIEW_REQUIRED,
                metadata={"reason": "inference_failed", "error_type": type(exc).__name__},
            )
