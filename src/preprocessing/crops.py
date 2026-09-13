from __future__ import annotations

import cv2
import numpy as np

from src.omr.template import TextQuestionRegion


def extract_text_crop(gray: np.ndarray, region: TextQuestionRegion) -> np.ndarray:
    """Extract one template-owned text ROI and trim only local whitespace.

    The page is never passed to an OCR provider. A small amount of local
    content tightening makes recognition models less sensitive to a very wide
    answer box while retaining the student's actual strokes unchanged.
    """
    box = region.bbox
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(gray.shape[1], box.x + box.width), min(gray.shape[0], box.y + box.height)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return crop

    margin_x = max(1, round(crop.shape[1] * region.ocr_inner_margin_x_ratio))
    margin_y = max(1, round(crop.shape[0] * region.ocr_inner_margin_y_ratio))
    inner = crop[margin_y:crop.shape[0] - margin_y, margin_x:crop.shape[1] - margin_x]
    if inner.size == 0:
        return crop

    # Use a local darkness mask only to find the content bounds. The original
    # grayscale pixels are returned to the OCR provider; no thresholded image
    # is sent as the recognition input.
    baseline = float(np.percentile(inner, 90))
    threshold = min(245.0, max(120.0, baseline - 25.0))
    mask = cv2.inRange(inner, 0, int(threshold))
    points = cv2.findNonZero(mask)
    if points is None:
        return inner
    bx, by, bw, bh = cv2.boundingRect(points)
    padding = max(4, round(min(inner.shape[:2]) * 0.04))
    bx0, by0 = max(0, bx - padding), max(0, by - padding)
    bx1, by1 = min(inner.shape[1], bx + bw + padding), min(inner.shape[0], by + bh + padding)
    return inner[by0:by1, bx0:bx1]
