# Minima

Minima is an in-progress W1.58A8 conversion of
[LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M)
for fast, low-memory CPU and CUDA inference without losing its 8,192-token context.
It includes packed kernels, quantization-aware distillation, recovery adapters,
task-tuning tools, and a spellchecker reference fine-tune.

> Release status: engineering validation. The public benchmark gate is 97% of a
> matched FP32 baseline. Results will be added from immutable HF Job outputs; no
> unmeasured speed or quality claims are made here.

## Why this is more than post-training rounding

BitNet b1.58 models use ternary matrix weights and int8 activations. Directly
rounding a conventionally trained encoder often loses too much quality, so Minima
uses teacher distillation and compact rank-128 recovery adapters around a packed ternary
backbone. The strict no-adapter profile remains available as an ablation.

- Logical matrix weights: `{-1, 0, +1}` (`log2(3) = 1.585` bits)
- Runtime storage: SIMD-friendly I2_S (four trits per byte)
- Activations: dynamic per-token/group int8
- CPU: fused AVX2 and ARM NEON dot-product extension
- CUDA: fused Triton unpack/matmul kernel
- Context: unchanged 8,192 tokens
- Tuning: STE QAT or lightweight recovery-adapter tuning

See [architecture](docs/architecture.md) and the exact [quality gates](docs/quality-gate.md).

## Install and use

```bash
pip install "minima-lfm @ git+https://github.com/SSHDotCodes/minima.git"
```

Once the gated checkpoint is published:

```python
from minima import MinimaModel

model = MinimaModel.from_pretrained("ProCreations/minima", device="cpu")
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
```

The first CPU use compiles a small PyTorch C++ extension and caches it. Set
`MINIMA_DISABLE_EXT=1` to force the portable reference implementation.

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

## License

Source code is MIT. Microsoft BitNet inspiration and attribution are in [NOTICE](NOTICE).
Derived LiquidAI weights remain under the LFM Open License v1.0, including its
commercial-use revenue threshold; see [MODEL_LICENSE](MODEL_LICENSE) and the full
license shipped with each model artifact.
