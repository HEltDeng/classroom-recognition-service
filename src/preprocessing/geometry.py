from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.omr.template import AnswerSheetTemplate, MarkerSpec
from src.preprocessing.quality import ImageQualityMetrics, measure_image_quality


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: str
    center: tuple[float, float]
    score: float


@dataclass(frozen=True)
class PageDetection:
    quad: np.ndarray | None
    perspective_quality: float
    page_complete: bool
    area_ratio: float
    details: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessResult:
    warped_gray: np.ndarray | None
    quality: ImageQualityMetrics
    page: PageDetection
    markers: list[MarkerDetection]
    errors: list[str]


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _quad_quality(quad: np.ndarray, image_area: float) -> tuple[float, float]:
    rect = _order_points(quad)
    area = abs(float(cv2.contourArea(rect.astype(np.float32))))
    area_ratio = min(1.0, max(0.0, area / max(image_area, 1.0)))
    cosines: list[float] = []
    for index in range(4):
        p0 = rect[index]
        p1 = rect[(index - 1) % 4] - p0
        p2 = rect[(index + 1) % 4] - p0
        denominator = float(np.linalg.norm(p1) * np.linalg.norm(p2))
        cosines.append(abs(float(np.dot(p1, p2) / denominator)) if denominator else 1.0)
    right_angle_score = 1.0 - min(1.0, max(cosines) / 0.5)
    return round(min(1.0, max(0.0, 0.5 * right_angle_score + 0.5 * area_ratio)), 4), area_ratio


def detect_page_quad(gray: np.ndarray) -> PageDetection:
    height, width = gray.shape[:2]
    image_area = float(height * width)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, np.ndarray, float, bool]] = []
    minimum_area = image_area * 0.18
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue
        quad = approximation.reshape(4, 2).astype(np.float32)
        quality, area_ratio = _quad_quality(quad, image_area)
        page_complete = bool(np.all(quad[:, 0] >= -3) and np.all(quad[:, 0] <= width + 3))
        page_complete = page_complete and bool(
            np.all(quad[:, 1] >= -3) and np.all(quad[:, 1] <= height + 3)
        )
        candidates.append((area, quad, quality, page_complete))

    if not candidates:
        return PageDetection(
            quad=None,
            perspective_quality=0.0,
            page_complete=False,
            area_ratio=0.0,
            details={"reason": "no_quadrilateral_detected"},
        )

    _, quad, quality, page_complete = max(candidates, key=lambda item: item[0])
    _, area_ratio = _quad_quality(quad, image_area)
    return PageDetection(
        quad=_order_points(quad),
        perspective_quality=quality,
        page_complete=page_complete and area_ratio >= 0.25,
        area_ratio=round(area_ratio, 4),
        details={"candidate_count": str(len(candidates))},
    )


def warp_quad(gray: np.ndarray, quad: np.ndarray, width: int, height: int) -> np.ndarray:
    source = _order_points(quad)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(gray, matrix, (width, height))


def _find_marker_in_region(gray: np.ndarray, marker: MarkerSpec) -> MarkerDetection | None:
    x0 = max(0, marker.x - marker.search_margin)
    y0 = max(0, marker.y - marker.search_margin)
    x1 = min(gray.shape[1], marker.x + marker.width + marker.search_margin)
    y1 = min(gray.shape[0], marker.y + marker.height + marker.search_margin)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = gray[y0:y1, x0:x1]
    _, binary = cv2.threshold(crop, 160, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expected_area = float(marker.width * marker.height)
    expected_center = np.array(marker.center, dtype=np.float32)
    best: tuple[float, MarkerDetection] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 3 or bh < 3 or area < expected_area * 0.18:
            continue
        center = np.array([x0 + bx + bw / 2, y0 + by + bh / 2], dtype=np.float32)
        distance = float(np.linalg.norm(center - expected_center))
        if distance > max(marker.search_margin * 1.6, 8):
            continue
        area_score = min(1.0, area / expected_area)
        aspect_score = min(bw, bh) / max(bw, bh)
        distance_score = max(0.0, 1.0 - distance / max(marker.search_margin * 1.6, 8))
        score = 0.45 * area_score + 0.25 * aspect_score + 0.30 * distance_score
        detection = MarkerDetection(marker.marker_id, (float(center[0]), float(center[1])), score)
        if best is None or score > best[0]:
            best = (score, detection)
    return best[1] if best else None


def detect_markers(gray: np.ndarray, template: AnswerSheetTemplate) -> list[MarkerDetection]:
    detections: list[MarkerDetection] = []
    for marker in template.markers:
        detection = _find_marker_in_region(gray, marker)
        if detection is not None:
            detections.append(detection)
    return detections


def refine_with_markers(
    warped_gray: np.ndarray,
    template: AnswerSheetTemplate,
    detections: list[MarkerDetection],
) -> np.ndarray:
    if len(detections) < 4:
        return warped_gray
    by_id = {item.marker_id: item for item in detections}
    source = np.array([by_id[item.marker_id].center for item in template.markers], dtype=np.float32)
    target = np.array([item.center for item in template.markers], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        warped_gray,
        matrix,
        (template.canonical_width, template.canonical_height),
    )


def preprocess_page(gray: np.ndarray, template: AnswerSheetTemplate) -> PreprocessResult:
    if gray.ndim != 2:
        raise ValueError("preprocess_page expects a grayscale image")
    quality = measure_image_quality(gray)
    page = detect_page_quad(gray)
    if page.quad is None:
        return PreprocessResult(
            warped_gray=None,
            quality=quality,
            page=page,
            markers=[],
            errors=["PAGE_NOT_FOUND"],
        )

    warped = warp_quad(gray, page.quad, template.canonical_width, template.canonical_height)
    markers = detect_markers(warped, template)
    errors: list[str] = []
    if quality.blur_score < template.quality_config.min_blur_score:
        errors.append("BLUR_TOO_HIGH")
    if quality.contrast < template.quality_config.min_contrast:
        errors.append("CONTRAST_TOO_LOW")
    if quality.brightness < template.quality_config.min_brightness:
        errors.append("IMAGE_TOO_DARK")
    if quality.brightness > template.quality_config.max_brightness:
        errors.append("IMAGE_TOO_BRIGHT")
    if template.quality_config.require_markers and len(markers) != len(template.markers):
        errors.append("MARKERS_NOT_FOUND")
    if not page.page_complete:
        errors.append("PAGE_INCOMPLETE")

    if not errors:
        warped = refine_with_markers(warped, template, markers)

    return PreprocessResult(
        warped_gray=warped,
        quality=quality,
        page=page,
        markers=markers,
        errors=errors,
    )
