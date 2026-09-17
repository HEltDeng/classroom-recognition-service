from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.config import ServiceSettings
from src.core.service import RecognitionService
from src.providers.registry import ProviderRegistry
from src.providers.unavailable import UnavailableTextRecognitionProvider
from src.schemas.models import RecognitionStatus
from tests.synthetic_factory import build_sheet


def make_service() -> RecognitionService:
    settings = ServiceSettings(default_ocr_provider="paddleocr")
    registry = ProviderRegistry(settings)
    registry._providers["paddleocr"] = UnavailableTextRecognitionProvider(
        name="test_ocr_unavailable", model_version="test", reason="test fixture"
    )
    return RecognitionService(
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
        settings=settings,
        provider_registry=registry,
    )


def choice_answers(response):
    return [answer for answer in response.answers if answer.type == "CHOICE"]


def test_perspective_sheet_returns_explainable_nominal_omr():
    response = make_service().recognize_array(
        build_sheet({1: "B", 2: "C", 3: "A", 4: "D"}, perspective=True)
    )
    choices = choice_answers(response)
    assert response.page_quality.status == "OK"
    assert [answer.value for answer in choices] == ["B", "C", "A", "D"]
    assert all(answer.status == RecognitionStatus.RECOGNIZED for answer in choices)
    assert all(set(answer.fill_ratios) == {"A", "B", "C", "D"} for answer in choices)
    assert all(0 <= answer.confidence <= 1 for answer in choices)
    assert response.engine_versions["omr_engine"] == "opencv-omr-v0.2"
    assert response.review_required is True  # OCR is deliberately unavailable in the base install.


def test_blank_sheet_is_explicitly_blank():
    response = make_service().recognize_array(build_sheet(perspective=True))
    choices = choice_answers(response)
    assert [answer.status for answer in choices] == [RecognitionStatus.BLANK] * 4
    assert all(answer.value is None for answer in choices)
    assert response.review_required is True


def test_multiple_mark_is_not_silently_resolved():
    response = make_service().recognize_array(
        build_sheet(multiple={1: ("A", "C")}, perspective=True)
    )
    answer = choice_answers(response)[0]
    assert answer.status == RecognitionStatus.MULTIPLE
    assert answer.value is None
    assert answer.fill_ratios["A"] > 0.28
    assert answer.fill_ratios["C"] > 0.28


def test_light_and_ambiguous_marks_need_review():
    service = make_service()
    light = choice_answers(service.recognize_array(build_sheet(light={1: "D"})))[0]
    ambiguous = choice_answers(
        service.recognize_array(build_sheet(ambiguous={1: ("A", "B")}))
    )[0]
    assert light.status == RecognitionStatus.TOO_LIGHT
    assert light.value is None
    assert ambiguous.status == RecognitionStatus.AMBIGUOUS
    assert ambiguous.value is None


def test_unusable_input_does_not_produce_answers():
    response = make_service().recognize_array(np.full((600, 800), 255, dtype="uint8"))
    assert response.page_quality.status == "PAGE_NOT_FOUND"
    assert response.answers == []
    assert response.review_required is True


def test_solid_black_mark_uses_surrounding_paper_as_reference():
    import cv2

    from src.omr.engine import OMRRecognitionEngine
    from src.omr.template import Rect
    image = np.full((100, 140), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 35), (90, 70), 0, -1)
    assert OMRRecognitionEngine()._measure_region(image, Rect(x=35,y=35,width=55,height=35), .18) > .95
