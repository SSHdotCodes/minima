# Architecture

Minima preserves the LFM2.5-Encoder-350M graph, including all 16 hybrid
short-convolution/attention layers and the 8,192-token supported context. It
changes the matrix representation and execution path:

1. Every dense projection and the token embedding use logical ternary values
   `{-1, 0, +1}` with per-output-channel group scales.
2. The hot `I2_S` representation packs four trits into each byte. Logical
   information density is `log2(3) = 1.585` bits; physical density is 2 bits so
   SIMD/GPU kernels can decode weights with shifts and masks.
3. Activations are dynamically quantized to signed int8 per token and group.
4. The quality profile adds rank-128 FP16 recovery adapters. These are deliberately
   small and tuneable; the strict profile omits them.
5. RMSNorm values, biases, and the 3-tap depthwise convolution remain FP16. They
   represent a tiny fraction of parameters and are not matrix-multiply weights.

The default CPU inference path fuses the ternary matrix and its recovery adapter
once, packs the effective matrix to per-channel INT8, releases the projection's
source tensors, and dispatches through PyTorch's oneDNN/FBGEMM dynamic GEMM. This
is the throughput profile. The strict I2_S extension instead fuses activation
quantization, 2-bit unpacking, and dot products directly; its AVX2 path uses
unsigned ternary codes and `maddubs`, while its ARM path uses NEON dot-product
instructions. The CUDA path fuses unpacking and matrix multiplication in Triton
so a full dequantized weight tensor is never materialized.

## Storage profiles

| Profile | Matrix weights | Activations | Recovery | Purpose |
|---|---:|---:|---:|---|
| strict | ternary I2_S | int8 | none | smallest artifact and ablation |
| quality | ternary I2_S | int8 | rank-128 FP16 | release candidate and tuning |

Measured sizes and resident-set memory are published only after the release jobs
complete. The original checkpoint is FP32, so its raw 354.5M parameters occupy
about 1.42 GB before framework overhead.
