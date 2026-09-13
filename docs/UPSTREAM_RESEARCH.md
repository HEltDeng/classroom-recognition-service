# Upstream research record

## OMRChecker

- Repository: [Udayraj123/OMRChecker](https://github.com/Udayraj123/OMRChecker)
- Local research path: `_research/upstream/OMRChecker/`
- Pinned checkout: `5cf44a5a9c7a49e2e541c6e19aefd09a41a4d494`
- Checkout branch: `master`
- HEAD message: `docs: fix typo noticible -> noticeable (#301)`
- License: MIT; the upstream `LICENSE` is preserved in the research checkout.

Read modules: `README.md`, `src/template.py`,
`src/schemas/template_schema.py`, `src/processors/CropPage.py`,
`src/processors/CropOnMarkers.py`, `src/core.py`, and `src/utils/image.py`.

### Reused ideas, not product structure

The formal `src/` implementation does not copy OMRChecker modules. It adapts
these traceable algorithm ideas:

1. JSON template-owned field geometry and grouped field blocks;
2. four-point perspective transform for page normalization;
3. marker-guided alignment as a fixed-sheet refinement step;
4. relative/local mark intensity comparison rather than one hard-coded answer
   threshold for every image;
5. explicit empty/multiple outcomes instead of forcing a value.

OMRChecker is primarily an image-directory CLI producing CSV/Excel-oriented
results. This service deliberately exposes an image + template version → JSON
boundary and keeps scoring outside the service.

## DeepSeek-OCR2

- Repository: [deepseek-ai/DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- Local research path: `_research/upstream/DeepSeek-OCR-2/`
- Pinned checkout: `2f3699ebbb96fa8af32212e8c170f2cc28730fad`
- Reference README says its tested environment is CUDA 11.8 + Torch 2.6 and
  uses GPU/bfloat16/FlashAttention-oriented inference.
- No model weights were downloaded by this project.
- Adapter status: conditional benchmark candidate only; not default.

## PaddleOCR

- Repository checked: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- Observed repository HEAD during project setup:
  `2661c7c0ef5c613e8f93c6e93b2e052399f0f854`
- Target recognition model setting: `PP-OCRv5_server_rec`; use
  `PP-OCRv5_mobile_rec` when the machine needs a smaller model.
- The integration is lazy and supports the recognition-only API when available,
  with a legacy pipeline compatibility path. It receives only a fixed text ROI,
  never the whole answer page.
