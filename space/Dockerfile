FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 gradio==5.49.1 "transformers>=4.56" huggingface-hub safetensors ninja \
    "minima-lfm @ git+https://github.com/SSHdotCodes/minima.git"

WORKDIR /app
COPY app.py /app/app.py
ENV MINIMA_MODEL=ProCreations/minima-spellcheck
ENV MINIMA_QUANTIZED_ENGINE=fbgemm
ENV OMP_NUM_THREADS=2
EXPOSE 7860
CMD ["python", "app.py"]
