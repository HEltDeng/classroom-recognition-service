from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np

from src.omr.template import AnswerSheetTemplate, TemplateRepository


def load_test_template() -> AnswerSheetTemplate:
    return TemplateRepository(Path(__file__).resolve().parents[1] / "templates").load("A-v0.1")


def build_sheet(
    answers: dict[int, str] | None = None,
    multiple: dict[int, tuple[str, str]] | None = None,
    light: dict[int, str] | None = None,
    ambiguous: dict[int, tuple[str, str]] | None = None,
    text_values: dict[int, str] | None = None,
    perspective: bool = True,
    blur: bool = False,
) -> np.ndarray:
    template = load_test_template()
    answers = answers or {}
    multiple = multiple or {}
    light = light or {}
    ambiguous = ambiguous or {}
    text_values = text_values or {}
    page = np.full((template.canonical_height, template.canonical_width), 255, dtype=np.uint8)

    # Page boundary and four fixed machine-vision markers.
    cv2.rectangle(page, (40, 40), (1560, 2160), 30, 8)
    for marker in template.markers:
        cv2.rectangle(
            page,
            (marker.x, marker.y),
            (marker.x + marker.width - 1, marker.y + marker.height - 1),
            0,
            -1,
        )

    for question in template.question_regions:
        cv2.putText(
            page,
            str(question.question_no),
            (300, question.options[0].bbox.y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            40,
            2,
            cv2.LINE_AA,
        )
        for option in question.options:
            box = option.bbox
            cv2.rectangle(
                page,
                (box.x, box.y),
                (box.x + box.width - 1, box.y + box.height - 1),
                80,
                3,
            )
            cv2.putText(
                page,
                option.label,
                (box.x + 45, box.y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                110,
                2,
                cv2.LINE_AA,
            )

        selected: list[tuple[str, int]] = []
        if question.question_no in answers:
            selected.append((answers[question.question_no], 0))
        if question.question_no in multiple:
            selected.extend(
                (label, 0 if index == 0 else 100)
                for index, label in enumerate(multiple[question.question_no])
            )
        if question.question_no in light:
            selected.append((light[question.question_no], 225))
        if question.question_no in ambiguous:
            selected.extend((label, 0) for label in ambiguous[question.question_no])
        for label, ink in selected:
            option = next(item for item in question.options if item.label == label)
            box = option.bbox
            cv2.rectangle(
                page,
                (box.x + 25, box.y + 18),
                (box.x + box.width - 26, box.y + box.height - 19),
                ink,
                -1,
            )

    for text_region in template.text_regions:
        box = text_region.bbox
        cv2.rectangle(
            page,
            (box.x, box.y),
            (box.x + box.width - 1, box.y + box.height - 1),
            90,
            3,
        )
        value = text_values.get(text_region.question_no)
        if value:
            cv2.putText(
                page,
                value,
                (box.x + 32, box.y + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                20,
                3,
                cv2.LINE_AA,
            )

    if not perspective:
        result = page
    else:
        canvas = np.full((2500, 2000), 255, dtype=np.uint8)
        source = np.array(
            [[0, 0], [template.canonical_width - 1, 0],
             [template.canonical_width - 1, template.canonical_height - 1], [0, template.canonical_height - 1]],
            dtype=np.float32,
        )
        destination = np.array(
            [[130, 130], [1870, 90], [1900, 2380], [90, 2410]], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        result = cv2.warpPerspective(page, matrix, (canvas.shape[1], canvas.shape[0]), borderValue=255)

    if blur:
        result = cv2.GaussianBlur(result, (21, 21), 0)
    return result


def to_png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode synthetic image")
    return io.BytesIO(encoded.tobytes()).getvalue()
