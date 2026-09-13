from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.config import ServiceSettings
from src.core.service import RecognitionService
from src.providers.registry import ProviderRegistry
from src.providers.unavailable import UnavailableTextRecognitionProvider
from tests.synthetic_factory import build_sheet, to_png_bytes


def test_http_upload_returns_standard_json_contract():
    service = make_service()
    client = TestClient(create_app(service))
    response = client.post(
        "/recognize/page",
        files={"image": ("answer-sheet.png", to_png_bytes(build_sheet({1: "B"})), "image/png")},
        data={"template_version": "A-v0.1", "page_type": "OMR", "page_token": "test-token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["template_version"] == "A-v0.1"
    assert body["page_type"] == "OMR"
    assert body["page_quality"]["status"] == "OK"
    assert body["answers"][0]["type"] == "CHOICE"
    assert body["answers"][0]["value"] == "B"
    assert "fill_ratios" in body["answers"][0]
    assert "threshold_config_version" in body["answers"][0]


def test_openapi_exposes_page_endpoint_and_health():
    service = make_service()
    client = TestClient(create_app(service))
    assert client.get("/health").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/recognize/page" in spec["paths"]
    assert "/health" in spec["paths"]


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
