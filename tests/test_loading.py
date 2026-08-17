import torch.nn as nn

from minima.loading import encoder_from_mlm


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.lfm2 = nn.Linear(3, 4)


def test_encoder_is_extracted_from_declared_mlm_wrapper():
    wrapper = Wrapper()
    assert encoder_from_mlm(wrapper) is wrapper.lfm2

