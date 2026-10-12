# Phase 1 eval harness: the app image (aiohttp, numpy, scipy, requests, the default voice) + parquet/audio/test deps.
FROM docker.io/speakrail/app:0.1.0
WORKDIR /eval
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import urllib.request as u; u.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz', '/opt/lid.176.ftz')"
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "cli.py"]
