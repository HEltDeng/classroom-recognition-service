# OCR Provider policy

All text engines implement `TextRecognitionProvider` and return the same
`TextRecognitionResult`. The business/service layer does not depend on a model
class.

| Provider | Role | Default | Requires |
|---|---|---:|---|
| `paddleocr` | PP-OCRv5 recognition-first primary | yes | `paddleocr`, `paddlepaddle` |
| `deepseek_vision` | low-confidence crop fallback | no | API key + model + network |
| `trocr` | benchmark candidate | no | Torch, Transformers, Pillow, model review |
| `deepseek_ocr2` | conditional GPU benchmark candidate | no | CUDA, Torch, Transformers, model weights |

PaddleOCR model initialization is lazy. A missing package, failed model load or
inference error returns `REVIEW_REQUIRED`, not a guessed string. The service can
therefore run OMR on a CPU-only/minimal installation.

`normalized_text` only strips surrounding whitespace. It does not correct
spelling, infer missing letters, use a standard answer, or turn `becuse` into
`because`.

DeepSeek Vision is disabled unless both `DEEPSEEK_API_KEY` and
`DEEPSEEK_VISION_MODEL` exist. Its request body contains one text ROI crop as a
data URL. It is never called with the whole page by the service. The prompt
requires actual written text, no correction/completion, and `uncertain` when
the crop is not legible. A production deployment still needs a privacy and
cost review before enabling it.
