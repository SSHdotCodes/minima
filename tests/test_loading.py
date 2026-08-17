import torch
import torch.nn as nn

from minima.loading import encoder_from_mlm
from minima.modeling import _materialize_derived_buffers


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.lfm2 = nn.Linear(3, 4)


def test_encoder_is_extracted_from_declared_mlm_wrapper():
    wrapper = Wrapper()
    assert encoder_from_mlm(wrapper) is wrapper.lfm2


def test_nonpersistent_rope_buffers_are_materialized():
    class Rotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = object()
            self.inv_freq = nn.Buffer(torch.empty(8, device="meta"), persistent=False)
            self.original_inv_freq = nn.Buffer(torch.empty(8, device="meta"), persistent=False)

        @staticmethod
        def compute_default_rope_parameters(config, device=None):
            return torch.arange(8, dtype=torch.float32, device=device), 1.0

    module = Rotary()
    _materialize_derived_buffers(module)
    assert module.inv_freq.device.type == "cpu"
    torch.testing.assert_close(module.inv_freq, torch.arange(8, dtype=torch.float32))
