# Validation report

Date: 2026-09-13 (Asia/Singapore)

## Scope

This report covers only image preprocessing, fixed-template OMR, OCR provider
integration, confidence/status output, and the Recognition Service API. No
scoring, student/organization business, dashboard, Android or Web project was
changed.

## Automated checks

- `python -m pytest -q`: **PASS**, 10 tests.
- `ruff check src tests benchmarks scripts`: **PASS**.
- `compileall`: **PASS**.
- Synthetic fixture generation: **PASS**.

OMR unit/HTTP fixtures cover nominal `1 B / 2 C / 3 A / 4 D`, blank, multiple,
too-light, ambiguous, page-not-found and explainable fill ratios. Non-
recognized states return `value: null` and set review.

## Real OMR HTTP smoke

The local Uvicorn process was started on `http://127.0.0.1:8092`, then stopped
after verification. The nominal fixture returned:

- page quality: `OK`;
- page detection and four markers: all detected;
- choices: `1B / 2C / 3A / 4D`;
- text: `because / environment / went`;
- `review_required: false`.

Separate HTTP uploads returned `BLANK`, `MULTIPLE`, `TOO_LIGHT`, and
`AMBIGUOUS` for their corresponding fixtures; none was converted to a guessed
choice.

## Real PaddleOCR Crop benchmark

Environment: Python 3.11.9, PaddleOCR 3.7.0, PaddlePaddle 3.3.1,
PaddleX 3.7.2, OpenCV contrib 4.10.0.84. The same preprocessed page and
template-owned text crops were used.

| Model | Exact match | CER | Mean latency | RSS observed | Status |
|---|---:|---:|---:|---:|---|
| `PP-OCRv5_server_rec` | 3/3 | 0.0 | 2658.87 ms | 539–544 MB | PASS on synthetic printed crops |
| `PP-OCRv5_mobile_rec` | 3/3 | 0.0 | 2585.87 ms | 413–416 MB | PASS on synthetic printed crops |

The first crop in each fresh process includes model initialization/download
cache overhead (server ~4934.88 ms, mobile ~7713.51 ms). Warm subsequent crops
were about 1477–1564 ms for server in the recorded run and about 21–23 ms for
mobile. These are one-host synthetic measurements, not a production SLA.

No-correction check: the server model received a synthetic `becuse` crop and
returned raw text `becuse` with Exact Match and CER 0 against that deliberately
misspelled ground truth.

Detailed raw records are in `benchmarks/results/`.

## Optional providers

- DeepSeek Vision: **not enabled**; no API key or model was configured. The
  provider is present and disabled by default, and its request implementation
  encodes only one crop.
- TrOCR: **PENDING_REAL_PROVIDER_RUN**; it remains benchmark-only pending model
  and data-license review.
- DeepSeek-OCR2: **PENDING_REAL_PROVIDER_RUN**; Torch/Transformers and model
  weights were not installed/downloaded. The host has an RTX 4060 Laptop GPU
  with 8188 MiB reported VRAM, below the adapter's conservative 10 GB gate.

## API evidence

- Runtime OpenAPI endpoint: `GET /openapi.json`.
- Static export: `docs/openapi.json`.
- Contract and request/response example: `docs/API_CONTRACT.md`.
