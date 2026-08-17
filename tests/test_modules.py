import torch
import torch.nn as nn

from minima.kernels.reference import reference_linear
from minima.modules import PackedTernaryEmbedding, PackedTernaryLinear, TernaryEmbedding, TernaryLinear


def test_packed_linear_matches_reference():
    torch.manual_seed(3)
    linear = nn.Linear(128, 19, bias=True)
    packed = PackedTernaryLinear.from_float(linear, group_size=128)
    x = torch.randn(5, 128)
    expected = reference_linear(x, packed.packed_weight, packed.weight_scale, 128, 128, packed.bias)
    actual = packed(x)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


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
