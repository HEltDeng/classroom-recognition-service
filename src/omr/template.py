from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Rect(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


class MarkerSpec(Rect):
    marker_id: str = Field(alias="id")
    search_margin: int = Field(default=32, ge=0)

    model_config = ConfigDict(populate_by_name=True)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


class OptionRegion(BaseModel):
    label: str
    bbox: Rect


class ChoiceQuestionRegion(BaseModel):
    question_no: int = Field(ge=1)
    type: Literal["CHOICE"] = "CHOICE"
    options: list[OptionRegion] = Field(min_length=2)


class TextQuestionRegion(BaseModel):
    question_no: int = Field(ge=1)
    type: Literal["TEXT"] = "TEXT"
    bbox: Rect
    ocr_inner_margin_x_ratio: float = Field(default=0.02, ge=0, lt=0.5)
    ocr_inner_margin_y_ratio: float = Field(default=0.08, ge=0, lt=0.5)


class OMRConfig(BaseModel):
    inner_margin_ratio: float = Field(default=0.18, ge=0, lt=0.5)
    blank_threshold: float = Field(default=0.06, ge=0, le=1)
    too_light_threshold: float = Field(default=0.18, ge=0, le=1)
    recognize_threshold: float = Field(default=0.28, ge=0, le=1)
    multiple_threshold: float = Field(default=0.28, ge=0, le=1)
    ambiguous_gap: float = Field(default=0.08, ge=0, le=1)
    minimum_gap: float = Field(default=0.10, ge=0, le=1)
    threshold_config_version: str = "omr-threshold-v0.1"


class QualityConfig(BaseModel):
    min_blur_score: float = Field(default=0.08, ge=0, le=1)
    min_contrast: float = Field(default=0.08, ge=0, le=1)
    min_brightness: float = Field(default=0.05, ge=0, le=1)
    max_brightness: float = Field(default=0.999, ge=0, le=1)
    require_markers: bool = True


class AnswerSheetTemplate(BaseModel):
    template_version: str
    page_type: Literal["OMR", "OCR"] = "OMR"
    canonical_width: int = Field(gt=0)
    canonical_height: int = Field(gt=0)
    markers: list[MarkerSpec] = Field(min_length=4)
    qr_region: Rect | None = None
    question_regions: list[ChoiceQuestionRegion] = Field(default_factory=list)
    text_regions: list[TextQuestionRegion] = Field(default_factory=list)
    omr_config: OMRConfig = Field(default_factory=OMRConfig)
    quality_config: QualityConfig = Field(default_factory=QualityConfig)
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.canonical_width, self.canonical_height


class TemplateNotFoundError(FileNotFoundError):
    pass


class TemplateRepository:
    def __init__(self, templates_dir: Path | None = None):
        self.templates_dir = templates_dir or Path(__file__).resolve().parents[2] / "templates"
        self._cache: dict[str, AnswerSheetTemplate] = {}

    def load(self, template_version: str) -> AnswerSheetTemplate:
        if not template_version or "/" in template_version or "\\" in template_version:
            raise TemplateNotFoundError(f"Invalid template version: {template_version!r}")
        if template_version in self._cache:
            return self._cache[template_version]
        path = self.templates_dir / f"{template_version}.json"
        if not path.is_file():
            registry_path = self.templates_dir / "registry.json"
            if registry_path.is_file():
                with registry_path.open("r", encoding="utf-8") as handle:
                    registry = json.load(handle)
                entry = next(
                    (
                        item
                        for item in registry.get("templates", [])
                        if item.get("template_version") == template_version
                    ),
                    None,
                )
                if entry and isinstance(entry.get("file"), str):
                    path = self.templates_dir / entry["file"]
        if not path.is_file():
            raise TemplateNotFoundError(f"Template not found: {template_version}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        template = AnswerSheetTemplate.model_validate(payload)
        if template.template_version != template_version:
            raise ValueError(
                f"Template filename/version mismatch: {path.name} vs {template.template_version}"
            )
        self._cache[template_version] = template
        return template

    def list_versions(self) -> list[str]:
        registry_path = self.templates_dir / "registry.json"
        if registry_path.is_file():
            with registry_path.open("r", encoding="utf-8") as handle:
                registry = json.load(handle)
            versions = [
                item.get("template_version")
                for item in registry.get("templates", [])
                if isinstance(item.get("template_version"), str)
            ]
            if versions:
                return sorted(versions)
        return sorted(path.stem for path in self.templates_dir.glob("*.json"))
