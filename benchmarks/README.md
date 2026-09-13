# OCR benchmark harness

`run_benchmark.py` preprocesses the same page, extracts the template-owned
fixed text ROI, and sends that crop to exactly one provider. It records:

- exact match and character error rate (CER);
- latency in milliseconds;
- provider confidence and status;
- optional process RSS/CPU metrics when `psutil` is installed;
- provider/model availability and metadata.

Example:

```powershell
python benchmarks/run_benchmark.py `
  --input datasets/synthetic/answer-sheet-a-v0.1-ocr-synthetic.png `
  --question 5 --truth because --provider paddleocr
```

For one provider instance across several fixed crops:

```powershell
python benchmarks/run_benchmark.py `
  --input datasets/synthetic/answer-sheet-a-v0.1-ocr-synthetic.png `
  --questions 5,6,7 --truths because,environment,went `
  --provider paddleocr
```

TrOCR and DeepSeek-OCR2 are benchmark candidates only. Missing model packages,
missing credentials, and insufficient GPU are recorded as unavailable/review
states, not as passing accuracy results.
