# Release results

All paid training and measurements ran as Hugging Face Jobs. The raw CPU and
quality reports are published in
[`ProCreations/minima-results`](https://huggingface.co/datasets/ProCreations/minima-results).

## Artifact and representation

| Checkpoint | Weight file | Change from source |
|---|---:|---:|
| LiquidAI source | 1,417,952,208 bytes | - |
| `ProCreations/minima` | 223,916,888 bytes | 84.2% smaller |
| `ProCreations/minima-spellcheck` | 226,027,664 bytes | 84.1% smaller |

The logical matrix alphabet is ternary, which carries `log2(3) = 1.585` bits of
information. I2_S physically stores four values per byte (2 bits/value) for
simple SIMD decoding. The quality artifact also contains rank-128 FP16 recovery
adapters, norms, biases, and depthwise convolution parameters.

## Encoder quality gate

Five tasks use matched 800-step schedules. CoLA reports the best of eight
predeclared Minima release-profile schedules and is therefore validation-tuned,
not a blind confirmatory result. Ratios are capped at 100% before averaging.

| Task | FP32 | Minima | Capped retention |
|---|---:|---:|---:|
| SST-2 | 0.93922 | 0.90596 | 96.46% |
| QNLI | 0.92239 | 0.90298 | 97.90% |
| MNLI | 0.84289 | 0.81100 | 96.22% |
| MRPC | 0.90780 | 0.91228 | 100.00% |
| STS-B | 0.90591 | 0.90307 | 99.69% |
| CoLA | 0.65358 | 0.58626 | 89.70% |
| **Mean** | | | **96.66%** |

The five non-CoLA tasks average 98.05% retention. CoLA selected 1,200 steps at
learning rate `1e-4`; every candidate is stored in `quality_gate.json`. The
composite **did not pass** the 97% release threshold, missing it by 0.34 percentage
points. The checkpoint is consequently labeled a release candidate rather than a
quality-gated release.

## CPU release gate

Hugging Face `cpu-performance`, Linux x86-64, FBGEMM, 16 threads, one warmup and
five measured runs:

| Sequence | FP32 median | Minima median | Speedup | FP32 peak RSS | Minima peak RSS | RSS reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 181.62 ms | 80.82 ms | 2.247x | 1,896.73 MB | 1,445.15 MB | 23.81% |
| 512 | 479.12 ms | 247.94 ms | 1.932x | 1,985.05 MB | 1,453.57 MB | 26.77% |
| 2,048 | 1,402.74 ms | 1,280.92 ms | 1.095x | 2,293.67 MB | 1,675.79 MB | 26.94% |
| 8,192 | 7,878.43 ms | 7,312.03 ms | 1.077x | 3,173.52 MB | 2,398.44 MB | 24.42% |

The default CPU throughput profile fuses ternary and recovery weights once,
packs per-channel dynamic INT8 with FBGEMM, and releases source projection
tensors. The strict `MINIMA_CPU_BACKEND=i2s` profile directly executes the packed
2-bit AVX2/ARM NEON representation but was not the fastest measured backend.

## GPU result

The fused Triton path is correct and avoids a full dequantized weight tensor, but
it did not beat the upstream BF16 implementation on an H200:

| Sequence | BF16 | Minima Triton |
|---:|---:|---:|
| 128 | 8.84 ms | 31.14 ms |
| 512 | 8.50 ms | 39.46 ms |
| 2,048 | 8.61 ms | 137.50 ms |
| 8,192 | 36.74 ms | 569.84 ms |

The CUDA code is therefore an optimization baseline and memory-oriented fused
path, not a GPU speed claim.

## Spellchecker validation

The 1,000-step packed tagger distillation used the LiquidAI spellchecker as
teacher with its optional dense reranker disabled on both sides.

| Diagnostic | Result |
|---|---:|
| Tag top-1 agreement within candidates | 99.51% |
| Error-detection top-1 agreement | 99.41% |
| Exact correction agreement | 70.0% (14/20) |
| Four published-style examples | 100% (4/4) |

These are teacher-agreement diagnostics, not an ERRANT benchmark. The live CPU
Space corrected `I has went to the stor yesterday .` to
`I went to the store yesterday.` in 110 ms after warmup.

## Compute budget

The launcher reserved each Job at its full timeout using the live hardware rate,
and refused launches beyond the project cap. The final ledger reserves $99.50 of
the $100 ceiling. Based on the last observable runtimes (16,520 H200 seconds and
612 `cpu-performance` seconds), estimated actual compute was **$23.27** at $5.00
and $1.90 per hour respectively. Billing may differ slightly because this is a
runtime-derived estimate rather than an account invoice.
