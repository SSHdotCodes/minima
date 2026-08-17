from __future__ import annotations

import types

import torch


def patch_tied_vocab_projection(model):
    """Teach the upstream GEC tagger to project through a packed embedding."""
    if not hasattr(model, "_label_logits") or not getattr(model.config, "tie_replace", False):
        return model

    def _label_logits(this, hidden):
        embedding = this.encoder.get_input_embeddings()
        base = this.base_head(hidden)
        replace_hidden = this.replace_proj(hidden)
        append_hidden = this.append_proj(hidden)
        if hasattr(embedding, "project"):
            rep = embedding.project(replace_hidden) + this.replace_bias
            app = embedding.project(append_hidden) + this.append_bias
        else:
            weight = embedding.weight[: this.vocab_size]
            rep = replace_hidden @ weight.t() + this.replace_bias
            app = append_hidden @ weight.t() + this.append_bias
        return torch.cat([base, rep, app], dim=-1)

    model._label_logits = types.MethodType(_label_logits, model)
    return model

