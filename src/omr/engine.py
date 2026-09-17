from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.omr.template import AnswerSheetTemplate, ChoiceQuestionRegion, Rect
from src.schemas.models import ChoiceAnswer, RecognitionStatus


@dataclass(frozen=True)
class OMRMeasurement:
    label: str
    fill_ratio: float


def _clamp(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


class OMRRecognitionEngine:
    """Fixed-template rectangle-mark OMR.

    The template owns every ROI. The engine only performs local image
    measurement and relative comparison; it never receives or uses an answer
    key, so scoring remains outside this service.
    """

    engine_version = "opencv-omr-v0.2"

    def _measure_region(self, gray: np.ndarray, rect: Rect, inner_margin_ratio: float) -> float:
        x, y, width, height = rect.as_xywh()
        mx = max(1, round(width * inner_margin_ratio))
        my = max(1, round(height * inner_margin_ratio))
        x0, y0 = x + mx, y + my
        x1, y1 = x + width - mx, y + height - my
        if x1 <= x0 or y1 <= y0:
            return 0.0
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
        # Estimate paper brightness outside the fill area: a completely filled
        # box has no white pixels of its own and must not become its own white.
        pad = max(4, height // 2)
        surrounding = gray[max(0, y-pad):min(gray.shape[0], y+height+pad),
                           max(0, x-pad):min(gray.shape[1], x+width+pad)]
        background = float(np.percentile(surrounding, 90))
        denominator = max(background - 20.0, 32.0)
        darkness = np.clip((background - crop.astype(np.float32)) / denominator, 0.0, 1.0)
        # Mean darkness is more useful than a single global threshold for
        # pencil/pen variation, while still behaving like an explainable ink
        # ratio for the fixed rectangular ROI.
        return round(_clamp(float(np.mean(darkness))), 4)

    def _confidence(
        self,
        status: RecognitionStatus,
        top_score: float,
        second_score: float,
        top_gap: float,
        template: AnswerSheetTemplate,
    ) -> float:
        config = template.omr_config
        if status == RecognitionStatus.RECOGNIZED:
            strength = _clamp(
                (top_score - config.recognize_threshold)
                / max(1.0 - config.recognize_threshold, 0.001)
            )
            separation = _clamp(top_gap / max(config.minimum_gap, 0.001))
            return round(_clamp(0.50 + 0.35 * strength + 0.15 * separation), 4)
        if status == RecognitionStatus.BLANK:
            return round(_clamp(1.0 - top_score / max(config.blank_threshold, 0.001)), 4)
        if status == RecognitionStatus.TOO_LIGHT:
            return round(_clamp(0.20 + 0.35 * top_score / max(config.too_light_threshold, 0.001)), 4)
        if status == RecognitionStatus.MULTIPLE:
            return round(_clamp(0.25 + 0.25 * top_score + 0.25 * second_score), 4)
        return round(_clamp(0.25 + 0.25 * top_score + 0.20 * top_gap), 4)

    def recognize_question(
        self,
        gray: np.ndarray,
        question: ChoiceQuestionRegion,
        template: AnswerSheetTemplate,
    ) -> ChoiceAnswer:
        measurements = [
            OMRMeasurement(
                label=option.label,
                fill_ratio=self._measure_region(
                    gray, option.bbox, template.omr_config.inner_margin_ratio
                ),
            )
            for option in question.options
        ]
        ordered = sorted(measurements, key=lambda item: item.fill_ratio, reverse=True)
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else OMRMeasurement("", 0.0)
        top_gap = round(max(0.0, top.fill_ratio - second.fill_ratio), 4)
        config = template.omr_config

        if top.fill_ratio < config.blank_threshold:
            status = RecognitionStatus.BLANK
            value = None
        elif top.fill_ratio < config.too_light_threshold:
            status = RecognitionStatus.TOO_LIGHT
            value = None
        elif second.fill_ratio >= config.multiple_threshold:
            status = (
                RecognitionStatus.AMBIGUOUS
                if top_gap <= config.ambiguous_gap
                else RecognitionStatus.MULTIPLE
            )
            value = None
        elif top.fill_ratio < config.recognize_threshold:
            status = RecognitionStatus.TOO_LIGHT
            value = None
        elif top_gap < config.minimum_gap:
            status = RecognitionStatus.AMBIGUOUS
            value = None
        else:
            status = RecognitionStatus.RECOGNIZED
            value = top.label

        return ChoiceAnswer(
            question_no=question.question_no,
            value=value,
            confidence=self._confidence(status, top.fill_ratio, second.fill_ratio, top_gap, template),
            status=status,
            fill_ratios={item.label: item.fill_ratio for item in measurements},
            engine=self.engine_version,
            threshold_config_version=config.threshold_config_version,
            top_score=top.fill_ratio,
            second_score=second.fill_ratio,
            top_gap=top_gap,
        )

    def recognize(self, gray: np.ndarray, template: AnswerSheetTemplate) -> list[ChoiceAnswer]:
        return [
            self.recognize_question(gray, question, template)
            for question in template.question_regions
        ]
