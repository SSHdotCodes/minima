import torch
import torch.nn as nn

from minima.kernels.reference import reference_linear
from minima.modeling import unpack_for_qat
from minima.modules import (
    PackedTernaryEmbedding,
    PackedTernaryLinear,
    TernaryEmbedding,
    TernaryLinear,
)
from minima.tuning import enable_recovery_training


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


def test_recovery_training_includes_small_non_matrix_parameters():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = PackedTernaryLinear.from_float(nn.Linear(32, 16), 32, 4)
            self.norm = nn.LayerNorm(16)

    model = ToyModel()
    count = enable_recovery_training(model)
    expected = sum(parameter.numel() for parameter in model.projection.parameters())
    expected += sum(parameter.numel() for parameter in model.norm.parameters())
    assert count == expected
    assert all(parameter.requires_grad for parameter in model.projection.parameters())
    assert all(parameter.requires_grad for parameter in model.norm.parameters())


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
    assert linear.log_scale.grad is not None
    assert linear.recovery_a.grad is not None

    embedding = TernaryEmbedding.from_float(nn.Embedding(13, 128), 128, recovery_rank=4)
    embedding(torch.tensor([[1, 2, 3]])).sum().backward()
    assert embedding.weight.grad is not None
    assert embedding.log_scale.grad is not None
    assert embedding.recovery_a.grad is not None


def test_qat_learned_scale_is_exported_without_recovery():
    torch.manual_seed(6)
    linear = TernaryLinear.from_float(nn.Linear(128, 17), group_size=128, recovery_rank=0)
    linear.log_scale.data.add_(0.2)
    packed = PackedTernaryLinear.from_float(linear, group_size=128)
    torch.testing.assert_close(packed.weight_scale, linear.log_scale.exp().half(), rtol=1e-3, atol=1e-5)
    assert packed.recovery_rank == 0


def test_qat_weight_curriculum_starts_from_dense_layer():
    torch.manual_seed(61)
    source = nn.Linear(128, 17)
    linear = TernaryLinear.from_float(
        source,
        group_size=64,
        recovery_rank=0,
        activation_quant=False,
    )
    linear.weight_quant_strength = 0.0
    x = torch.randn(3, 128)
    torch.testing.assert_close(linear(x), source(x))
    linear.weight_quant_strength = 1.0
    assert not torch.allclose(linear(x), source(x))


def test_strict_packed_model_can_expand_for_full_weight_qat():
    torch.manual_seed(7)
    packed = PackedTernaryLinear.from_float(nn.Linear(128, 17), group_size=128)
    model = nn.Sequential(packed)
    x = torch.randn(3, 128)
    expected = model(x)
    unpack_for_qat(model)
    assert isinstance(model[0], TernaryLinear)
    assert model[0].recovery_rank == 0
    torch.testing.assert_close(model(x), expected, rtol=2e-3, atol=2e-3)
