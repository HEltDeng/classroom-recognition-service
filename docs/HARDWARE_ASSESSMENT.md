# Hardware assessment

Read-only host check on 2026-09-13:

| Resource | Observed |
|---|---|
| CPU | Intel Core i9-13980HX, 24 cores / 32 logical processors |
| RAM | 15.63 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8188 MiB reported by `nvidia-smi` |
| Driver | 581.57 |
| Python ML packages | Torch, PaddleOCR and PaddlePaddle were not installed at the initial audit |

Conclusion: OpenCV/NumPy OMR is suitable for local CPU execution. PP-OCRv5
should be benchmarked in CPU/mobile mode first, then optionally with the
available GPU. DeepSeek-OCR2 is not selected as a default: its upstream
instructions are GPU-oriented and this host has only 8 GB VRAM with no Torch
runtime installed at audit time. The adapter applies a conservative 10 GB VRAM
availability gate and records the candidate as unavailable when that gate is
not met.

This is a hardware suitability assessment, not a successful DeepSeek-OCR2
inference result. A real model run remains `PENDING_REAL_PROVIDER_RUN` until a
deliberate benchmark environment is provisioned.
