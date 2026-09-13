# Synthetic fixtures

These fixtures are generated, non-PII images for the `A-v0.1` technical
validation template. They intentionally cover nominal, blank, multiple-mark,
too-light, ambiguous, printed-text, and deliberate misspelling cases.

Generate or refresh them with:

```powershell
python scripts/generate_synthetic_samples.py
```

Do not place identifiable student images in this directory. Private local
material belongs in `datasets/local-private/`, which is ignored by Git.
