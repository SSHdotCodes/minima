from __future__ import annotations

import os
import re
import time

import gradio as gr

from minima import MinimaModel

MODEL_ID = os.environ.get("MINIMA_MODEL", "ProCreations/minima-spellcheck")
model = MinimaModel.from_pretrained(MODEL_ID, device="cpu")


def whitespace_tokenize(text: str) -> str:
    return " ".join(re.findall(r"\w+(?:['’]\w+)?|[^\w\s]", text, flags=re.UNICODE))


def detokenize(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\b(n)\s+'\s+(t)\b", r"\1'\2", text, flags=re.IGNORECASE)
    return text


def correct(text: str, precision: float, passes: int):
    if not text.strip():
        return "", "Enter text to check."
    prepared = whitespace_tokenize(text[:8000])
    start = time.perf_counter()
    result = model.correct([prepared], min_error_prob=precision, max_iter=int(passes))[0]
    elapsed = 1000 * (time.perf_counter() - start)
    return detokenize(result), f"{elapsed:.0f} ms · packed W1.58A8 CPU inference"


with gr.Blocks(title="Minima Spellcheck", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown(
        "# Minima Spellcheck\n"
        "Grammar, spelling, punctuation, and casing with a packed ternary LFM2.5 encoder. "
        "The model runs entirely on this Space's CPU."
    )
    source = gr.Textbox(label="Text", lines=8, value="I has went to the stor yesterday.")
    with gr.Row():
        precision = gr.Slider(0.0, 0.95, value=0.35, step=0.05, label="Minimum error probability")
        passes = gr.Slider(1, 4, value=3, step=1, label="Refinement passes")
    button = gr.Button("Check text", variant="primary")
    output = gr.Textbox(label="Correction", lines=8)
    status = gr.Markdown()
    button.click(correct, [source, precision, passes], [output, status])
    gr.Examples(
        examples=[
            ["She go to school every day."],
            ["Their are many reason to study hard."],
            ["Him and me was late for the meetting."],
            ["That's a fair point, let's discuss it tomorrow."],
        ],
        inputs=[source],
    )
    gr.Markdown(
        "[Model](https://huggingface.co/ProCreations/minima-spellcheck) · "
        "[Source](https://github.com/SSHdotCodes/minima) · 8,192-token encoder context"
    )

demo.queue(default_concurrency_limit=2).launch(server_name="0.0.0.0", server_port=7860)

