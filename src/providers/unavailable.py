from __future__ import annotations

import numpy as np

from src.providers.base import TextRecognitionProvider, TextRecognitionResult
from src.schemas.models import RecognitionStatus


class UnavailableTextRecognitionProvider(TextRecognitionProvider):
    def __init__(self, name: str, model_version: str, reason: str):
        self.name = name
        self.model_version = model_version
        self.reason = reason

    @property
    def is_available(self) -> bool:
        return False

    def recognize_crop(self, crop: np.ndarray) -> TextRecognitionResult:
        del crop
        return TextRecognitionResult(
            raw_text=None,
            confidence=0.0,
            engine=self.name,
            status=RecognitionStatus.REVIEW_REQUIRED,
            metadata={"reason": self.reason},
        )
