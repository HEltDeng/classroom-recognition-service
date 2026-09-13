from __future__ import annotations

from src.core.config import ServiceSettings
from src.providers.base import TextRecognitionProvider
from src.providers.deepseek_ocr2 import DeepSeekOCR2BenchmarkProvider
from src.providers.deepseek_vision import DeepSeekVisionApiProvider
from src.providers.paddleocr_provider import PaddleOcrProvider
from src.providers.trocr import TrOCRBenchmarkProvider


class ProviderRegistry:
    def __init__(self, settings: ServiceSettings | None = None):
        self.settings = settings or ServiceSettings.from_env()
        self._providers: dict[str, TextRecognitionProvider] = {
            "paddleocr": PaddleOcrProvider(
                model_name=self.settings.paddleocr_model_name,
                device=self.settings.paddleocr_device,
            ),
            "deepseek_vision": DeepSeekVisionApiProvider(
                api_base=self.settings.deepseek_api_base,
                model=self.settings.deepseek_vision_model,
            ),
            "trocr": TrOCRBenchmarkProvider(),
            "deepseek_ocr2": DeepSeekOCR2BenchmarkProvider(),
        }

    def get(self, name: str | None = None) -> TextRecognitionProvider:
        requested = (name or self.settings.default_ocr_provider).lower()
        try:
            return self._providers[requested]
        except KeyError as exc:
            raise ValueError(f"Unknown OCR provider: {requested}") from exc

    def fallback_providers(self, primary_name: str | None = None) -> list[TextRecognitionProvider]:
        primary = (primary_name or self.settings.default_ocr_provider).lower()
        # The Vision API is the only automatic fallback. Benchmark candidates
        # must be explicitly invoked by the benchmark harness.
        candidate = self._providers["deepseek_vision"]
        return [candidate] if primary != "deepseek_vision" and candidate.is_available else []

    def metadata(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, provider in self._providers.items():
            result[f"provider_{key}_available"] = str(provider.is_available).lower()
            result[f"provider_{key}_model"] = provider.model_version
        return result
