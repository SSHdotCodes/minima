# Quality and release gates

No Minima checkpoint is called a release until it passes all gates below.

## Encoder quality

The primary score is a matched, single-seed fine-tune comparison on six tasks
covering classification, sentence pairs, similarity, and linguistic acceptability:
SST-2, QNLI, MNLI, MRPC, STS-B, and CoLA. The FP32 teacher and Minima use identical
data, heads, steps, and seeds. Per-task ratios are capped at 1.0 before averaging
so improvements cannot hide a regression. Required relative mean: **>= 0.97**.

The full upstream 17-task protocol is much larger than the project compute cap;
the six-task gate is therefore identified explicitly rather than presented as an
upstream-table reproduction. Representation cosine, masked-token agreement, and
8k-context smoke tests are diagnostics, not substitutes for the downstream gate.

## Performance

- CPU wall time must improve over the upstream FP32 model at sequence lengths
  128, 512, 2,048, and 8,192 on the same host and thread count.
- Peak resident memory must be lower, and artifact bytes are reported separately.
- CUDA correctness is checked against the reference kernel before GPU timing.
- Every number includes hardware, software versions, warmup count, and raw JSON.

## Spellchecker

The packed spellchecker is compared with LiquidAI's FP16 tagger under the same
iterative decode settings with its optional dense reranker disabled for both
models. The report includes held-out tag and detection agreement, exact correction
agreement, example outputs, and a clean-text false-positive check. The reranker is
excluded because it contains a separate 1.42 GB FP32 encoder and would invalidate
the packed CPU memory measurement. Public ERRANT scores, when reported, are kept
separate from teacher-agreement diagnostics.
