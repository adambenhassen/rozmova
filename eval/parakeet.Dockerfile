# Parakeet v3 for the Gate A comparison (offline + NeMo buffered streaming), GPU.
FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir "nemo_toolkit[asr]==2.7.3" soundfile requests "num2words>=0.5.14,<0.6" && mkdir -p /nemo && \
    python -c "import urllib.request as u; u.urlretrieve('https://raw.githubusercontent.com/NVIDIA/NeMo/v2.7.3/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py', '/nemo/speech_to_text_streaming_infer_rnnt.py')"
WORKDIR /eval
ENV PYTHONUNBUFFERED=1 HF_HOME=/hf
