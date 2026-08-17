"""Minima: W1.58A8 LFM2.5 encoder inference and task tuning."""

from .modeling import MinimaModel
from .modules import PackedTernaryEmbedding, PackedTernaryLinear, TernaryEmbedding, TernaryLinear
from .quantization import QuantizedWeight, dequantize, pack_i2s, quantize_ternary, unpack_i2s

__all__ = [
    "MinimaModel",
    "PackedTernaryEmbedding",
    "PackedTernaryLinear",
    "QuantizedWeight",
    "TernaryEmbedding",
    "TernaryLinear",
    "dequantize",
    "pack_i2s",
    "quantize_ternary",
    "unpack_i2s",
]

__version__ = "0.1.0"

