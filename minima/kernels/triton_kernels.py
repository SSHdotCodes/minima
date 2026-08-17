"""Fused I2_S unpack + matrix multiplication for NVIDIA GPUs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _i2s_gemm(a, packed, scales, out, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
               groups: tl.constexpr, quarter: tl.constexpr, group_size: tl.constexpr,
               stride_am: tl.constexpr, stride_ak: tl.constexpr,
               stride_pn: tl.constexpr, stride_pg: tl.constexpr, stride_pq: tl.constexpr,
               stride_sn: tl.constexpr, stride_sg: tl.constexpr,
               stride_om: tl.constexpr, stride_on: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        kk = k0 + offs_k
        group = kk // group_size
        within = kk % group_size
        qindex = within % quarter
        lane = within // quarter
        a_tile = tl.load(a + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
                         mask=(offs_m[:, None] < m) & (kk[None, :] < k), other=0.0)
        bytes_ = tl.load(
            packed + offs_n[:, None] * stride_pn + group[None, :] * stride_pg + qindex[None, :] * stride_pq,
            mask=(offs_n[:, None] < n) & (kk[None, :] < k), other=1,
        )
        codes = (bytes_ >> (lane[None, :] * 2)) & 3
        scale = tl.load(scales + offs_n[:, None] * stride_sn + group[None, :] * stride_sg,
                        mask=(offs_n[:, None] < n) & (group[None, :] < groups), other=0.0)
        w = (codes.to(tl.float32) - 1.0) * scale
        acc += tl.dot(a_tile, tl.trans(w.to(a_tile.dtype)))

    tl.store(out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def triton_i2s_linear(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, in_features: int,
                      group_size: int) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("Triton kernel requires a CUDA tensor")
    x = x.contiguous()
    packed = packed.to(device=x.device, non_blocking=True).contiguous()
    scale = scale.to(device=x.device, dtype=torch.float32, non_blocking=True).contiguous()
    m, k = x.shape
    n, groups, quarter = packed.shape
    if k != in_features or quarter * 4 != group_size:
        raise ValueError("incompatible input or packed-weight shape")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    # A full quantization group per K tile amortizes scale loads and lets
    # tl.dot feed tensor cores with substantially larger tiles. Smaller M
    # tiles avoid wasting work for short, single-sequence encoder calls.
    block_m = 32 if m < 256 else 64
    block_n, block_k = 64, 128
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _i2s_gemm[grid](
        x, packed, scale, out, m, n, k, groups, quarter, group_size,
        x.stride(0), x.stride(1), packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=8, num_stages=3,
    )
    return out
