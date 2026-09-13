from __future__ import annotations

import base64
import json
import os
from typing import Any

import cv2
import numpy as np

from src.providers.base import TextRecognitionProvider, TextRecognitionResult
from src.schemas.models import RecognitionStatus

DEEPSEEK_VISION_PROMPT = (
    "只识别图片中学生实际写出的英文文本。不要纠正拼写，不要根据常识补全，"
    "不要根据正确答案推测。看不清时返回 uncertain。只返回 JSON："
    '{"raw_text": string|null, "confidence": number, "status": "recognized"|"uncertain"}。'
)


class DeepSeekVisionApiProvider(TextRecognitionProvider):
    """Optional OpenAI-compatible DeepSeek Vision fallback.

    The provider is disabled until both DEEPSEEK_API_KEY and
    DEEPSEEK_VISION_MODEL are configured. The request body contains only the
    individual crop passed to recognize_crop.
    """

    name = "deepseek_vision"

    def __init__(
        self,
        api_base: str = "https://api.deepseek.com",
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_version = model or "unconfigured"
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.enabled = bool(self._api_key and model)

    @property
    def is_available(self) -> bool:
        if not self.enabled:
            return False
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _encode_crop(crop: np.ndarray) -> str:
        if crop.ndim == 2:
            image = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            image = crop
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise ValueError("could not encode OCR crop")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _parse_content(content: Any) -> dict[str, Any]:
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            raise TypeError("vision response content is not text")
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise TypeError("vision response is not an object")
        return parsed

    def recognize_crop(self, crop: np.ndarray) -> TextRecognitionResult:
        if not self.is_available:
            return TextRecognitionResult(
                raw_text=None,
                confidence=0.0,
                engine=self.name,
                status=RecognitionStatus.REVIEW_REQUIRED,
                metadata={"reason": "api_key_or_model_not_configured"},
            )
        try:
            import httpx

            image_data = self._encode_crop(crop)
            payload = {
                "model": self.model_version,
                "temperature": 0,
                "max_tokens": 128,
                "messages": [
                    {"role": "system", "content": DEEPSEEK_VISION_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "识别这个单独的答题 ROI。"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    },
                ],
            }
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=45.0,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = self._parse_content(content)
            raw_text = parsed.get("raw_text")
            raw_text = str(raw_text) if raw_text is not None else None
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.0))))
            uncertain = str(parsed.get("status", "uncertain")).lower() != "recognized"
            status = (
                RecognitionStatus.RECOGNIZED
                if raw_text and not uncertain
                else RecognitionStatus.REVIEW_REQUIRED
            )
            return TextRecognitionResult(
                raw_text=raw_text,
                confidence=round(confidence, 4),
                engine=self.name,
                status=status,
                metadata={"model": self.model_version},
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary converts failures to review
            return TextRecognitionResult(
                raw_text=None,
                confidence=0.0,
                engine=self.name,
                status=RecognitionStatus.REVIEW_REQUIRED,
                metadata={"reason": "vision_request_failed", "error_type": type(exc).__name__},
            )

    def version_metadata(self) -> dict[str, str]:
        return {
            "ocr_provider": self.name,
            "ocr_model": self.model_version,
            "ocr_enabled": str(self.enabled).lower(),
        }
