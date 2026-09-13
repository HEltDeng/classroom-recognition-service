from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQualityMetrics:
    blur_score: float
    brightness: float
    contrast: float


def _clamp(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def measure_image_quality(gray: np.ndarray) -> ImageQualityMetrics:
    if gray.ndim != 2:
        raise ValueError("measure_image_quality expects a grayscale image")
    if gray.size == 0:
        return ImageQualityMetrics(blur_score=0.0, brightness=0.0, contrast=0.0)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian_variance = float(cv2.Laplacian(blurred, cv2.CV_64F).var())
    # The score is deliberately bounded and monotonic rather than pretending to
    # be a universal camera-quality probability.
    blur_score = _clamp(1.0 - math.exp(-max(0.0, laplacian_variance) / 180.0))
    brightness = _clamp(float(np.mean(gray)) / 255.0)
    contrast = _clamp(float(np.std(gray)) / 64.0)
    return ImageQualityMetrics(
        blur_score=round(blur_score, 4),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
    )
