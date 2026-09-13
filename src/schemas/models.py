from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecognitionStatus(str, Enum):
    RECOGNIZED = "RECOGNIZED"
    BLANK = "BLANK"
    MULTIPLE = "MULTIPLE"
    AMBIGUOUS = "AMBIGUOUS"
    TOO_LIGHT = "TOO_LIGHT"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class PageQualityStatus(str, Enum):
    OK = "OK"
    IMAGE_RETAKE_REQUIRED = "IMAGE_RETAKE_REQUIRED"
    PAGE_NOT_FOUND = "PAGE_NOT_FOUND"
    MARKERS_NOT_FOUND = "MARKERS_NOT_FOUND"


class PageQuality(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    status: PageQualityStatus
    blur_score: float | None = Field(default=None, ge=0, le=1)
    brightness: float | None = Field(default=None, ge=0, le=1)
    contrast: float | None = Field(default=None, ge=0, le=1)
    perspective_quality: float | None = Field(default=None, ge=0, le=1)
    page_complete: bool = False
    marker_score: float | None = Field(default=None, ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    question_no: int = Field(ge=1)
    type: Literal["CHOICE"] = "CHOICE"
    value: str | None = None
    confidence: float = Field(ge=0, le=1)
    status: RecognitionStatus
    fill_ratios: dict[str, float]
    engine: str = "opencv_omr"
    threshold_config_version: str
    top_score: float = Field(ge=0, le=1)
    second_score: float = Field(ge=0, le=1)
    top_gap: float = Field(ge=0, le=1)

    @field_validator("fill_ratios")
    @classmethod
    def validate_fill_ratios(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("fill_ratios must not be empty")
        return {str(key): min(1.0, max(0.0, float(ratio))) for key, ratio in value.items()}


class TextAnswer(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    question_no: int = Field(ge=1)
    type: Literal["TEXT"] = "TEXT"
    raw_text: str | None = None
    normalized_text: str | None = None
    confidence: float = Field(ge=0, le=1)
    engine: str
    status: RecognitionStatus
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("normalized_text")
    @classmethod
    def strip_without_correcting(cls, value: str | None) -> str | None:
        # Normalization is deliberately limited to surrounding whitespace. In
        # particular, no spell correction or language-model completion is done.
        return value.strip() if value is not None else None


Answer = Annotated[ChoiceAnswer | TextAnswer, Field(discriminator="type")]


class RecognitionResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    request_id: str
    template_version: str
    page_type: Literal["OMR"] = "OMR"
    page_quality: PageQuality
    answers: list[Answer] = Field(default_factory=list)
    review_required: bool = False
    engine_versions: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
