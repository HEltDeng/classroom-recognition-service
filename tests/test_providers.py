from __future__ import annotations

import numpy as np

from src.providers.deepseek_vision import DEEPSEEK_VISION_PROMPT, DeepSeekVisionApiProvider
from src.providers.paddleocr_provider import PaddleOcrProvider
from src.schemas.models import RecognitionStatus


def test_vision_provider_is_disabled_without_credentials_and_preserves_prompt_boundary():
    provider = DeepSeekVisionApiProvider(model=None, api_key=None)
    result = provider.recognize_crop(np.full((40, 120), 255, dtype=np.uint8))
    assert provider.is_available is False
    assert result.status == RecognitionStatus.REVIEW_REQUIRED
    assert result.raw_text is None
    assert "不要纠正拼写" in DEEPSEEK_VISION_PROMPT
    assert "不要根据正确答案推测" in DEEPSEEK_VISION_PROMPT


def test_paddle_provider_has_lazy_and_explicit_review_behavior():
    provider = PaddleOcrProvider()
    if provider.is_available:
        # Package presence is not an inference pass. Avoid downloading a model
        # during unit tests; the real provider run is an explicit benchmark.
        assert provider.version_metadata()["ocr_provider"] == "paddleocr_ppocrv5"
    else:
        result = provider.recognize_crop(np.full((40, 120), 255, dtype=np.uint8))
        assert result.status == RecognitionStatus.REVIEW_REQUIRED
        assert result.raw_text is None
