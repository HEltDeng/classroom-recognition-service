# Benchmark plan

The benchmark harness uses the same preprocessed page and the same template
fixed text crop for each provider. It records exact match, CER, latency,
confidence, status, availability, and optional CPU/RSS metrics.

Required corpus dimensions:

- normal English words;
- connected handwriting, case variants, pencil, black pen;
- light writing, heavy writing, erasure, blur, tilt and shadow;
- deliberately misspelled text such as `becuse` to verify no correction.

Current repository fixtures are synthetic and printed, not handwriting and not
a production accuracy claim. Real identifiable student images belong only in
`datasets/local-private/` and are not committed.

Provider result classification:

- `PASS` only when the provider actually ran on the crop and the metric was
  produced;
- `PENDING_REAL_PROVIDER_RUN` when dependencies/model/GPU/credentials are
  unavailable;
- `REVIEW_REQUIRED` for a provider's uncertain result;
- never convert an unavailable model into an accuracy score.
