# Minima

Minima is a W1.58A8 release candidate converted from
[LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M)
for fast, low-memory CPU inference and fused CUDA kernel research without losing
its 8,192-token context. It includes packed kernels, quantization-aware
distillation, recovery adapters, task-tuning tools, and a spellchecker reference
fine-tune. The current CUDA path is memory-oriented and does not beat BF16 H200
latency; exact timings are published rather than presented as a speedup.

> Release status: **candidate**. It retained 96.66% on the declared six-task
> downstream gate, missing the 97% target by 0.34 percentage points. CPU speed,
> memory, spellchecker, and all quality results are measured and public.

## Why this is more than post-training rounding

BitNet b1.58 models use ternary matrix weights and int8 activations. Directly
rounding a conventionally trained encoder often loses too much quality, so Minima
uses teacher distillation and compact rank-128 recovery adapters around a packed ternary
backbone. The strict no-adapter profile remains available as an ablation.

- Logical matrix weights: `{-1, 0, +1}` (`log2(3) = 1.585` bits)
- Runtime storage: SIMD-friendly I2_S (four trits per byte)
- Activations: dynamic per-token/group int8
- CPU: one-time ternary+recovery fusion into oneDNN/FBGEMM dynamic INT8 GEMMs;
  fused AVX2 and ARM NEON I2_S kernels remain available as the strict backend
- CUDA: fused Triton unpack/matmul kernel
- Context: unchanged 8,192 tokens
- Tuning: STE QAT or lightweight recovery-adapter tuning

See [architecture](docs/architecture.md), the exact [quality gates](docs/quality-gate.md),
and the [measured release results](docs/results.md).

## Install and use

```bash
pip install "minima-lfm @ git+https://github.com/SSHDotCodes/minima.git"
```

The public candidate checkpoint is available now:

```python
from minima import MinimaModel

model = MinimaModel.from_pretrained("ProCreations/minima", device="cpu")
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
```

CPU inference defaults to the fast dynamic-INT8 backend. It expands the ternary
artifact once, fuses each recovery adapter into its matrix, packs the result with
per-channel INT8 scales, and releases the source projection tensors. This keeps
runtime memory far below FP32 while using the platform's optimized GEMM library.
Set `MINIMA_CPU_BACKEND=i2s` for direct 2-bit AVX2/NEON execution; its first use
compiles and caches the small C++ extension. Set `MINIMA_DISABLE_EXT=1` to force
the portable reference implementation.

## Convert, distill, and benchmark

```bash
minima-convert LiquidAI/LFM2.5-Encoder-350M ./minima-ptq --group-size 128

minima-train \
  --steps 4000 --sequence-length 512 --group-size 32 --recovery-rank 128 \
  --output-repo ProCreations/minima

minima-benchmark ProCreations/minima \
  --lengths 128,512,2048,8192 --output results/cpu.json
```

Heavy conversion, training, and evaluation run as Hugging Face Jobs. The committed
launcher reserves worst-case cost from live HF hardware prices and refuses any
launch that would put this project above **$100**:

```bash
python scripts/hf_job.py --name qat --flavor h200 --timeout 6h -- \
  bash -lc 'git clone --depth 1 https://github.com/SSHDotCodes/minima /src/minima && ...'
```

The reservation ledger is [hf_jobs_ledger.json](hf_jobs_ledger.json). Unrelated
Jobs in the same account are not modified or charged to this project ledger.

## Published artifacts

- [Base candidate](https://huggingface.co/ProCreations/minima)
- [Spellchecker](https://huggingface.co/ProCreations/minima-spellcheck)
- [Live spellchecker Space](https://huggingface.co/spaces/ProCreations/minima-spellcheck)
- [Immutable benchmark reports](https://huggingface.co/datasets/ProCreations/minima-results)

## License

Source code is MIT. Microsoft BitNet inspiration and attribution are in [NOTICE](NOTICE).
Derived LiquidAI weights remain under the LFM Open License v1.0, including its
commercial-use revenue threshold; see [MODEL_LICENSE](MODEL_LICENSE) and the full
license shipped with each model artifact.
