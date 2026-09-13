from __future__ import annotations

import importlib.metadata
import json
from typing import Any

import cv2
import numpy as np

from src.providers.base import TextRecognitionProvider, TextRecognitionResult
from src.schemas.models import RecognitionStatus


class PaddleOcrProvider(TextRecognitionProvider):
    """PP-OCRv5 recognition-only adapter.

    PaddleOCR has had more than one Python API shape. This adapter first tries
    the recognition-only API and then supports the older pipeline API as a
    compatibility fallback. It never sends the whole page to PaddleOCR.
    """

    name = "paddleocr_ppocrv5"

    def __init__(self, model_name: str = "PP-OCRv5_server_rec", device: str = "cpu"):
        self.model_version = model_name
        self.device = device
        self._client: Any | None = None
        self._mode: str | None = None
        self._load_error: str | None = None
        try:
            self.package_version = importlib.metadata.version("paddleocr")
        except importlib.metadata.PackageNotFoundError:
            self.package_version = "not-installed"

    @property
    def is_available(self) -> bool:
        return self.package_version != "not-installed"

    def _load(self) -> None:
        if self._client is not None or self._load_error is not None:
            return
        if not self.is_available:
            self._load_error = "paddleocr package is not installed"
            return
        try:
            import paddleocr

            recognition_cls = getattr(paddleocr, "TextRecognition", None)
            if recognition_cls is not None:
                try:
                    self._client = recognition_cls(
                        model_name=self.model_version,
                        device=self.device,
                    )
                except TypeError:
                    self._client = recognition_cls(model_name=self.model_version)
                self._mode = "recognition"
            else:
                pipeline_cls = getattr(paddleocr, "PaddleOCR", None)
                if pipeline_cls is None:
                    raise RuntimeError("PaddleOCR.TextRecognition/PaddleOCR API not found")
                try:
                    self._client = pipeline_cls(
                        lang="en",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        device=self.device,
                    )
                except TypeError:
                    self._client = pipeline_cls(lang="en", use_angle_cls=False, show_log=False)
                self._mode = "pipeline"
        except Exception as exc:  # noqa: BLE001 - provider boundary converts failures to review
            self._load_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _json_value(value: Any) -> Any:
        if callable(value):
            value = value()
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @classmethod
    def _parse_v3_result(cls, result: Any) -> tuple[list[str], list[float]]:
        data = cls._json_value(getattr(result, "json", result))
        if isinstance(data, dict) and isinstance(data.get("res"), dict):
            data = data["res"]
        if not isinstance(data, dict):
            return [], []
        texts = data.get("rec_texts") or data.get("texts") or []
        scores = data.get("rec_scores") or data.get("scores") or []
        if not texts and data.get("rec_text") is not None:
            texts = [data["rec_text"]]
        if not scores and data.get("rec_score") is not None:
            scores = [data["rec_score"]]
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [str(text) for text in texts], [float(score) for score in scores]

    @staticmethod
    def _parse_legacy_result(raw: Any) -> tuple[list[str], list[float]]:
        if not raw:
            return [], []
        rows = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
        texts: list[str] = []
        scores: list[float] = []
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            value = row[1]
            if isinstance(value, (list, tuple)) and value:
                texts.append(str(value[0]))
                scores.append(float(value[1]) if len(value) > 1 else 0.0)
        return texts, scores

    def recognize_crop(self, crop: np.ndarray) -> TextRecognitionResult:
        self._load()
        if self._client is None:
            return TextRecognitionResult(
                raw_text=None,
                confidence=0.0,
                engine=self.name,
                status=RecognitionStatus.REVIEW_REQUIRED,
                metadata={"reason": self._load_error or "provider_unavailable"},
            )
        if crop.ndim == 2:
            input_image = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            input_image = crop
        try:
            if self._mode == "recognition":
                raw_results = self._client.predict(input=input_image)
                texts: list[str] = []
                scores: list[float] = []
                for result in raw_results or []:
                    parsed_texts, parsed_scores = self._parse_v3_result(result)
                    texts.extend(parsed_texts)
                    scores.extend(parsed_scores)
            else:
                raw_results = self._client.ocr(input_image, cls=False)
                texts, scores = self._parse_legacy_result(raw_results)
        except Exception as exc:  # noqa: BLE001 - provider boundary converts failures to review
            return TextRecognitionResult(
                raw_text=None,
                confidence=0.0,
                engine=self.name,
                status=RecognitionStatus.REVIEW_REQUIRED,
                metadata={"reason": "inference_failed", "error_type": type(exc).__name__},
            )

        raw_text = " ".join(texts).strip() if texts else None
        confidence = min(1.0, max(0.0, float(np.mean(scores)))) if scores else 0.0
        return TextRecognitionResult(
            raw_text=raw_text,
            confidence=round(confidence, 4),
            engine=self.name,
            status=(RecognitionStatus.RECOGNIZED if raw_text else RecognitionStatus.REVIEW_REQUIRED),
            metadata={"package_version": self.package_version, "model": self.model_version},
        )

    def version_metadata(self) -> dict[str, str]:
        return {
            "ocr_provider": self.name,
            "ocr_provider_version": self.package_version,
            "ocr_model": self.model_version,
            "ocr_mode": self._mode or "lazy",
        }
