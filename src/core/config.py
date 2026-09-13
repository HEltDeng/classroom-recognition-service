from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceSettings:
    default_template_version: str = "A-v0.1"
    default_ocr_provider: str = "paddleocr"
    max_upload_bytes: int = 12 * 1024 * 1024
    ocr_fallback_min_confidence: float = 0.75
    paddleocr_model_name: str = "PP-OCRv5_server_rec"
    paddleocr_device: str = "cpu"
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_vision_model: str | None = None

    @classmethod
    def from_env(cls) -> ServiceSettings:
        return cls(
            default_template_version=os.getenv("RECOGNITION_TEMPLATE_VERSION", "A-v0.1"),
            default_ocr_provider=os.getenv("OCR_PROVIDER", "paddleocr").lower(),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024))),
            ocr_fallback_min_confidence=float(os.getenv("OCR_FALLBACK_MIN_CONFIDENCE", "0.75")),
            paddleocr_model_name=os.getenv("PADDLEOCR_MODEL_NAME", "PP-OCRv5_server_rec"),
            paddleocr_device=os.getenv("PADDLEOCR_DEVICE", "cpu"),
            deepseek_api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            deepseek_vision_model=os.getenv("DEEPSEEK_VISION_MODEL") or None,
        )
