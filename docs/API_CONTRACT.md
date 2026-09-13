# API Contract

## `POST /recognize/page`

`multipart/form-data`：

| Field | Required | Description |
|---|---:|---|
| `image` | yes | One PNG/JPEG page image |
| `template_version` | no | Defaults to `A-v0.1` |
| `page_type` | no | Defaults to `OMR` |
| `page_token` | no | Optional correlation token; not interpreted by recognition |

The service does not accept an answer key. It returns recognition only.

### Example request

```bash
curl -X POST http://127.0.0.1:8092/recognize/page \
  -F 'image=@datasets/synthetic/answer-sheet-a-v0.1-recognized.png;type=image/png' \
  -F 'template_version=A-v0.1' \
  -F 'page_type=OMR'
```

### Example response

```json
{
  "request_id": "4d0d8baf-9f54-4a8a-9948-0bbd9ef1d602",
  "template_version": "A-v0.1",
  "page_type": "OMR",
  "page_quality": {
    "status": "OK",
    "blur_score": 0.991,
    "brightness": 0.947,
    "contrast": 0.633,
    "perspective_quality": 0.842,
    "page_complete": true,
    "marker_score": 0.985,
    "details": {
      "preprocessing_errors": [],
      "detected_markers": ["top_left", "top_right", "bottom_left", "bottom_right"]
    }
  },
  "answers": [
    {
      "question_no": 1,
      "type": "CHOICE",
      "value": "B",
      "confidence": 0.97,
      "status": "RECOGNIZED",
      "fill_ratios": {"A": 0.01, "B": 0.81, "C": 0.02, "D": 0.01},
      "engine": "opencv-omr-v0.1",
      "threshold_config_version": "omr-threshold-v0.1",
      "top_score": 0.81,
      "second_score": 0.02,
      "top_gap": 0.79
    },
    {
      "question_no": 5,
      "type": "TEXT",
      "raw_text": "environment",
      "normalized_text": "environment",
      "confidence": 0.91,
      "engine": "paddleocr_ppocrv5",
      "status": "RECOGNIZED",
      "provider_metadata": {"model": "PP-OCRv5_server_rec"}
    }
  ],
  "review_required": false,
  "engine_versions": {
    "recognition_service": "0.1.0",
    "omr_engine": "opencv-omr-v0.1",
    "template_version": "A-v0.1",
    "threshold_config_version": "omr-threshold-v0.1",
    "ocr_provider": "paddleocr_ppocrv5",
    "ocr_model": "PP-OCRv5_server_rec"
  },
  "errors": []
}
```

If the page is unusable, `answers` is empty and `review_required` is `true`.
If a choice is blank, multiple, too light, or ambiguous, its `value` is `null`.
If OCR is not installed/configured, text answers are `REVIEW_REQUIRED`; OMR is
not blocked.

## Status semantics

`RECOGNIZED`, `BLANK`, `MULTIPLE`, `AMBIGUOUS`, `TOO_LIGHT`,
`LOW_IMAGE_QUALITY`, and `REVIEW_REQUIRED` are recognition states. They do not
mean correct/incorrect and carry no scoring decision.
