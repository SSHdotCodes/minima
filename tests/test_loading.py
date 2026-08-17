import torch
import torch.nn as nn

from minima.loading import encoder_from_mlm
from minima.modeling import _materialize_derived_buffers, _transcode_packed_state
from minima.quantization import pack_i2s


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


def test_base3_checkpoint_state_expands_to_i2s_runtime():
    trits = torch.randint(-1, 2, (3, 64), dtype=torch.int8)
    runtime = pack_i2s(trits, 32)
    metadata = {
        "storage_format": "base3",
        "modules": {"projection": {"group_size": 32}},
    }
    stored = _transcode_packed_state(
        {"projection.packed_weight": runtime.clone()}, metadata, loading=False,
    )
    assert stored["projection.packed_weight"].shape == (3, 2, 7)
    restored = _transcode_packed_state(stored, metadata, loading=True)
    torch.testing.assert_close(restored["projection.packed_weight"], runtime)


def test_cuda_artifacts_use_one_floating_runtime_dtype(monkeypatch, tmp_path):
    """FP32 QAT residual tensors must not promote an FP16 packed stream."""
    from minima import modeling

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4, dtype=torch.float32)
            self.register_buffer("codes", torch.ones(4, dtype=torch.uint8))
            self.config = object()

        def get_input_embeddings(self):
            return None

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "minima_config.json").write_text(
        '{"format_version": 1, "base_model": "fake", "base_revision": null, '
        '"model_kind": "encoder", "modules": {}}'
    )
    state = FakeModel().state_dict()

    monkeypatch.setattr(modeling.AutoConfig, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(modeling, "build_lfm_encoder", lambda config: FakeModel())
    monkeypatch.setattr(modeling, "load_file", lambda *args, **kwargs: state)
    monkeypatch.setattr(modeling, "_materialize_derived_buffers", lambda model: None)

    original_to = FakeModel.to

    def record_cuda_to(self, *args, **kwargs):
        # Exercise dtype conversion without requiring a CUDA runner in CI.
        assert kwargs["device"] == torch.device("cuda")
        return original_to(self, dtype=kwargs["dtype"])

    monkeypatch.setattr(FakeModel, "to", record_cuda_to)
    loaded = modeling.MinimaModel.from_pretrained(artifact, device="cuda")
    assert loaded.model.norm.weight.dtype == torch.float16
    assert loaded.model.codes.dtype == torch.uint8
