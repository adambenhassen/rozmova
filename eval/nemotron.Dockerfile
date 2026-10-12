# Nemotron 3.5 ASR streaming (cache-aware FastConformer-RNNT, 40 locales incl. uk-UA) for the Phase 2 ASR comparison.
# NeMo is pinned to the main commit that ships the model's prompt support and the cache-aware streaming script.
FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# gcc: Triton compiles kernels at runtime for the RNNT decoder
ARG NEMO_SHA=6677a40b7530d2d7b263728c45e9723de183a8e8
RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 ffmpeg gcc libc6-dev && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir Cython packaging && \
    pip install --no-cache-dir "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@${NEMO_SHA}" soundfile requests \
        "num2words>=0.5.14,<0.6" "fasttext-predict>=0.9.2.4,<0.10" && \
    mkdir -p /nemo && python -c "import urllib.request as u; \
u.urlretrieve('https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/${NEMO_SHA}/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py', '/nemo/cache_aware_streaming_infer.py'); \
u.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz', '/opt/lid.176.ftz')"
WORKDIR /eval
ENV PYTHONUNBUFFERED=1 HF_HOME=/hf
