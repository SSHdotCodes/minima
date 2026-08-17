import torch
import torch.nn as nn

from minima.kernels.reference import reference_linear
from minima.modules import (
    PackedTernaryEmbedding,
    PackedTernaryLinear,
    TernaryEmbedding,
    TernaryLinear,
    optimize_cpu_model,
)


def test_packed_linear_matches_reference():
    torch.manual_seed(3)
    linear = nn.Linear(128, 19, bias=True)
    packed = PackedTernaryLinear.from_float(linear, group_size=128).eval()
    x = torch.randn(5, 128)
    expected = reference_linear(x, packed.packed_weight, packed.weight_scale, 128, 128, packed.bias)
    actual = packed(x)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_group32_packed_linear_matches_reference():
    torch.manual_seed(30)
    linear = nn.Linear(128, 23, bias=False)
    packed = PackedTernaryLinear.from_float(linear, group_size=32).eval()
    x = torch.randn(7, 128)
    expected = reference_linear(x, packed.packed_weight, packed.weight_scale, 128, 32)
    torch.testing.assert_close(packed(x), expected, rtol=2e-3, atol=2e-3)


def test_packed_recovery_adapter_has_end_to_end_gradients():
    torch.manual_seed(31)
    packed = PackedTernaryLinear.from_float(nn.Linear(128, 19), group_size=128, recovery_rank=4).train()
    x = torch.randn(2, 128, requires_grad=True)
    packed(x).square().mean().backward()
    assert x.grad is not None
    assert packed.recovery_a.grad is not None
    assert packed.recovery_b.grad is not None


def test_dynamic_int8_cpu_fuses_ternary_and_recovery():
    if not [engine for engine in torch.backends.quantized.supported_engines if engine != "none"]:
        return
    torch.manual_seed(32)
    packed = PackedTernaryLinear.from_float(
        nn.Linear(128, 19), group_size=32, recovery_rank=4,
    ).eval()
    x = torch.randn(7, 128)
    expected = packed(x)
    packed.optimize_cpu(release_source=True)
    actual = packed(x)
    relative_mae = (actual - expected).abs().mean() / expected.abs().mean().clamp_min(1.0e-6)
    assert relative_mae < 0.03
    assert packed.packed_weight.numel() == 0


def test_dynamic_int8_cpu_fuses_gated_mlp_pair():
    if not [engine for engine in torch.backends.quantized.supported_engines if engine != "none"]:
        return

    class GatedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = PackedTernaryLinear.from_float(nn.Linear(32, 64, bias=False), 32, 4)
            self.w3 = PackedTernaryLinear.from_float(nn.Linear(32, 64, bias=False), 32, 4)
            self.w2 = PackedTernaryLinear.from_float(nn.Linear(64, 32, bias=False), 32, 4)

        def forward(self, x):
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    torch.manual_seed(33)
    model = GatedMLP().eval()
    x = torch.randn(5, 32)
    expected = model(x)
    optimize_cpu_model(model)
    actual = model(x)
    relative_mae = (actual - expected).abs().mean() / expected.abs().mean().clamp_min(1.0e-6)
    assert relative_mae < 0.06
    assert model.w1.packed_weight.numel() == 0
    assert model.w3.packed_weight.numel() == 0


def test_packed_embedding_matches_dequantized_rows():
    torch.manual_seed(4)
    embedding = nn.Embedding(31, 128, padding_idx=0)
    packed = PackedTernaryEmbedding.from_float(embedding, group_size=128)
    ids = torch.tensor([[0, 2, 9], [4, 2, 30]])
    actual = packed(ids)
    expected = packed._rows(ids, actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_qat_modules_backpropagate():
    torch.manual_seed(5)
    linear = TernaryLinear.from_float(nn.Linear(128, 17), group_size=128, recovery_rank=4)
    x = torch.randn(3, 128, requires_grad=True)
    linear(x).square().mean().backward()
    assert x.grad is not None
    assert linear.weight.grad is not None
    assert linear.recovery_a.grad is not None

    embedding = TernaryEmbedding.from_float(nn.Embedding(13, 128), 128, recovery_rank=4)
    embedding(torch.tensor([[1, 2, 3]])).sum().backward()
    assert embedding.weight.grad is not None
    assert embedding.recovery_a.grad is not None
