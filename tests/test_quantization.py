import torch

from minima.quantization import (
    QuantizedWeight,
    dequantize,
    fake_quantize_weight,
    pack_i2s,
    quantize_ternary,
    unpack_i2s,
)


def test_i2s_roundtrip_multiple_groups():
    torch.manual_seed(1)
    values = torch.randint(-1, 2, (7, 256), dtype=torch.int8)
    packed = pack_i2s(values, 128)
    assert packed.shape == (7, 2, 32)
    torch.testing.assert_close(unpack_i2s(packed, 256, 128), values)


def test_quantization_padding_and_dequantization():
    torch.manual_seed(2)
    weight = torch.randn(11, 130)
    quant = quantize_ternary(weight, 128)
    assert quant.packed.shape == (11, 2, 32)
    assert quant.scale.shape == (11, 2)
    restored = dequantize(quant)
    assert restored.shape == weight.shape
    assert set(restored.sign().unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_dequantize_known_values():
    trits = torch.tensor([[-1, 0, 1, -1] * 8], dtype=torch.int8)
    packed = pack_i2s(trits, 32)
    quant = QuantizedWeight(packed, torch.tensor([[0.5]], dtype=torch.float16), (1, 32), 32)
    torch.testing.assert_close(dequantize(quant), trits.float() * 0.5)


def test_explicit_group_scale_has_gradients_and_survives_export():
    torch.manual_seed(8)
    weight = torch.randn(3, 128, requires_grad=True)
    scale = torch.full((3, 1), 0.25, requires_grad=True)
    fake_quantize_weight(weight, 128, scale).square().mean().backward()
    assert weight.grad is not None
    assert scale.grad is not None
    quant = quantize_ternary(weight, 128, scale)
    torch.testing.assert_close(quant.scale, scale.detach().half())
