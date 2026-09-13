from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.schemas.models import RecognitionStatus


@dataclass(frozen=True)
class TextRecognitionResult:
    raw_text: str | None
    confidence: float
    engine: str
    status: RecognitionStatus
    metadata: dict[str, Any] = field(default_factory=dict)


class TextRecognitionProvider(ABC):
    name: str = "unknown"
    model_version: str = "unknown"

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the provider can be used without changing configuration."""

    @abstractmethod
    def recognize_crop(self, crop: np.ndarray) -> TextRecognitionResult:
        """Recognize only the supplied fixed text ROI crop."""

    def version_metadata(self) -> dict[str, str]:
        return {
            "ocr_provider": self.name,
            "ocr_model": self.model_version,
        }
