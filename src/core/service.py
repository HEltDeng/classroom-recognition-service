from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from src.core.config import ServiceSettings
from src.omr.engine import OMRRecognitionEngine
from src.omr.template import TemplateRepository, TextQuestionRegion
from src.preprocessing.crops import extract_text_crop
from src.preprocessing.geometry import PreprocessResult, preprocess_page
from src.providers.base import TextRecognitionProvider, TextRecognitionResult
from src.providers.registry import ProviderRegistry
from src.schemas.models import (
    Answer,
    PageQuality,
    PageQualityStatus,
    RecognitionResponse,
    RecognitionStatus,
    TextAnswer,
)


class InvalidImageError(ValueError):
    pass


class RecognitionService:
    def __init__(
        self,
        templates_dir: Path | None = None,
        settings: ServiceSettings | None = None,
        provider_registry: ProviderRegistry | None = None,
    ):
        self.settings = settings or ServiceSettings.from_env()
        self.templates = TemplateRepository(templates_dir)
        self.omr = OMRRecognitionEngine()
        self.providers = provider_registry or ProviderRegistry(self.settings)

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise InvalidImageError("image is empty")
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise InvalidImageError("image could not be decoded")
        return image

    def _page_quality(self, processed: PreprocessResult) -> PageQuality:
        errors = processed.errors
        if "PAGE_NOT_FOUND" in errors:
            status = PageQualityStatus.PAGE_NOT_FOUND
        elif any(
            error in errors
            for error in (
                "BLUR_TOO_HIGH",
                "CONTRAST_TOO_LOW",
                "IMAGE_TOO_DARK",
                "IMAGE_TOO_BRIGHT",
                "PAGE_INCOMPLETE",
            )
        ):
            status = PageQualityStatus.IMAGE_RETAKE_REQUIRED
        elif "MARKERS_NOT_FOUND" in errors:
            status = PageQualityStatus.MARKERS_NOT_FOUND
        else:
            status = PageQualityStatus.OK
        return PageQuality(
            status=status,
            blur_score=processed.quality.blur_score,
            brightness=processed.quality.brightness,
            contrast=processed.quality.contrast,
            perspective_quality=processed.page.perspective_quality,
            page_complete=processed.page.page_complete,
            marker_score=(
                round(float(np.mean([marker.score for marker in processed.markers])), 4)
                if processed.markers
                else None
            ),
            details={
                "preprocessing_errors": errors,
                "detected_markers": [marker.marker_id for marker in processed.markers],
                "page_area_ratio": processed.page.area_ratio,
                **processed.page.details,
            },
        )

    @staticmethod
    def _text_answer(
        question_no: int,
        result: TextRecognitionResult,
    ) -> TextAnswer:
        raw_text = result.raw_text
        return TextAnswer(
            question_no=question_no,
            raw_text=raw_text,
            normalized_text=raw_text.strip() if raw_text is not None else None,
            confidence=result.confidence,
            engine=result.engine,
            status=result.status,
            provider_metadata=result.metadata,
        )

    def _recognize_text_region(
        self,
        gray: np.ndarray,
        region: TextQuestionRegion,
    ) -> tuple[TextAnswer, dict[str, str]]:
        crop = extract_text_crop(gray, region)
        return self._recognize_prepared_text_region(crop, region.question_no)

    def _recognize_prepared_text_region(
        self,
        crop: np.ndarray,
        question_no: int,
    ) -> tuple[TextAnswer, dict[str, str]]:
        primary = self.providers.get()
        result = primary.recognize_crop(crop)
        selected_provider: TextRecognitionProvider = primary

        if result.status != RecognitionStatus.RECOGNIZED or (
            result.confidence < self.settings.ocr_fallback_min_confidence
        ):
            for fallback in self.providers.fallback_providers(primary.name):
                fallback_result = fallback.recognize_crop(crop)
                if fallback_result.confidence > result.confidence or (
                    result.status != RecognitionStatus.RECOGNIZED
                    and fallback_result.status == RecognitionStatus.RECOGNIZED
                ):
                    result = fallback_result
                    selected_provider = fallback

        metadata = selected_provider.version_metadata()
        return self._text_answer(question_no, result), metadata

    def recognize_array(
        self,
        image: np.ndarray,
        template_version: str | None = None,
        page_type: str = "OMR",
        request_id: str | None = None,
    ) -> RecognitionResponse:
        request_id = request_id or str(uuid.uuid4())
        template_version = template_version or self.settings.default_template_version
        template = self.templates.load(template_version)
        if page_type != template.page_type:
            raise ValueError(
                f"page_type {page_type!r} does not match template page_type {template.page_type!r}"
            )
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2:
            raise InvalidImageError("image must be grayscale or BGR")

        processed = preprocess_page(image, template)
        page_quality = self._page_quality(processed)
        engine_versions = {
            "recognition_service": "0.1.0",
            "omr_engine": self.omr.engine_version,
            "template_version": template.template_version,
            "threshold_config_version": template.omr_config.threshold_config_version,
            **self.providers.metadata(),
        }
        if processed.warped_gray is None or page_quality.status != PageQualityStatus.OK:
            return RecognitionResponse(
                request_id=request_id,
                template_version=template.template_version,
                page_quality=page_quality,
                answers=[],
                review_required=True,
                engine_versions=engine_versions,
                errors=processed.errors,
            )

        answers: list[Answer] = list(self.omr.recognize(processed.warped_gray, template))
        primary = self.providers.get()
        engine_versions["ocr_provider"] = primary.name
        engine_versions["ocr_model"] = primary.model_version
        for text_region in template.text_regions:
            text_answer, provider_metadata = self._recognize_text_region(
                processed.warped_gray, text_region
            )
            answers.append(text_answer)
            engine_versions.update(provider_metadata)

        review_required = any(answer.status != RecognitionStatus.RECOGNIZED for answer in answers)
        return RecognitionResponse(
            request_id=request_id,
            template_version=template.template_version,
            page_quality=page_quality,
            answers=answers,
            review_required=review_required,
            engine_versions=engine_versions,
            errors=[],
        )

    def recognize_bytes(
        self,
        image_bytes: bytes,
        template_version: str | None = None,
        page_type: str = "OMR",
        request_id: str | None = None,
    ) -> RecognitionResponse:
        return self.recognize_array(
            self._decode_image(image_bytes),
            template_version=template_version,
            page_type=page_type,
            request_id=request_id,
        )
