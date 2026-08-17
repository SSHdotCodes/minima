"""Version-robust loading helpers for the remote-code LFM2.5 encoder."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForMaskedLM


def encoder_from_mlm(wrapper):
    """Extract the body while preserving the checkpoint's `lfm2.` key prefix.

    Some Transformers releases instantiate the `AutoModel` class directly and
    fail to strip the MLM checkpoint's `lfm2.` prefix, silently leaving random
    body weights. Loading the declared MLM architecture first avoids that bug.
    """
    for attribute in ("lfm2", "model", "base_model"):
        body = getattr(wrapper, attribute, None)
        if body is not None and body is not wrapper:
            return body
    raise AttributeError("could not locate the LFM2 encoder body in the masked-LM wrapper")


def load_lfm_encoder(model_id: str, **kwargs: Any):
    wrapper = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    return encoder_from_mlm(wrapper)


def build_lfm_encoder(config):
    wrapper = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    return encoder_from_mlm(wrapper)

