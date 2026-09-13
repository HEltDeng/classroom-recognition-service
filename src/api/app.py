from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.core.service import InvalidImageError, RecognitionService
from src.omr.template import TemplateNotFoundError
from src.schemas.models import RecognitionResponse


def create_app(service: RecognitionService | None = None) -> FastAPI:
    recognition_service = service or RecognitionService()
    app = FastAPI(
        title="Classroom Recognition Service",
        description=(
            "Fixed-template page preprocessing, rectangle OMR, and replaceable OCR. "
            "This service does not score answers or manage classroom data."
        ),
        version="0.1.0",
    )
    app.state.recognition_service = recognition_service

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "classroom-recognition-service",
            "version": "0.1.0",
            "templates": recognition_service.templates.list_versions(),
            "ocr": recognition_service.providers.metadata(),
        }

    @app.get("/templates")
    def templates() -> dict[str, list[str]]:
        return {"template_versions": recognition_service.templates.list_versions()}

    @app.post("/recognize/page", response_model=RecognitionResponse)
    async def recognize_page(
        image: Annotated[UploadFile, File(description="One answer-sheet image")],
        template_version: Annotated[str, Form()] = "A-v0.1",
        page_type: Annotated[str, Form()] = "OMR",
        page_token: Annotated[str | None, Form()] = None,
    ) -> RecognitionResponse:
        del page_token  # reserved for upstream correlation; not used for recognition
        payload = await image.read()
        if len(payload) > recognition_service.settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="image exceeds MAX_UPLOAD_BYTES")
        try:
            return recognition_service.recognize_bytes(
                payload,
                template_version=template_version,
                page_type=page_type,
            )
        except TemplateNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidImageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
