#!/usr/bin/env bash
# Run the eval on the Speakrail compose network; the eval dir is mounted, so data/, sets/ and out/ persist here.
#   ./run.sh build-sets | run --sets uk_read ... | tts --adapter breeze | summary | test | parakeet offline|stream [set...] | nemotron <lang> <left,right> [set...]
set -euo pipefail
cd "$(dirname "$0")"
NET=${SPEAKRAIL_NET:-speakrail_default}
docker build -q -t speakrail-eval . >/dev/null
case "${1:-}" in
  test) docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/eval" --entrypoint pytest speakrail-eval -q -p no:cacheprovider ;;
  parakeet)
    shift
    docker build -q -t speakrail-eval-parakeet -f parakeet.Dockerfile . >/dev/null
    mkdir -p "$HOME/.cache/huggingface"
    docker run --rm --gpus device=${PARAKEET_GPU:-1} --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -e PARAKEET_CHUNK_S -e PARAKEET_RIGHT_S -e PARAKEET_BATCH -e PARAKEET_LIMIT -e PARAKEET_TAG -v "$PWD:/eval" \
      -v "$HOME/.cache/huggingface:/hf" speakrail-eval-parakeet python parakeet.py "$@" ;;
  nemotron)
    shift
    docker build -q -t speakrail-eval-nemotron -f nemotron.Dockerfile . >/dev/null
    mkdir -p "$HOME/.cache/huggingface"
    # USER/TORCHINDUCTOR_CACHE_DIR: torch looks the uid up in /etc/passwd otherwise, and the host uid has no entry
    docker run --rm --gpus device=${NEMOTRON_GPU:-1} --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=eval \
      -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -v "$PWD:/eval" \
      -v "$HOME/.cache/huggingface:/hf" speakrail-eval-nemotron python nemotron.py "$@" ;;
  *) docker run --rm --init --user "$(id -u):$(id -g)" --network "$NET" -v "$PWD:/eval" speakrail-eval "$@" ;;
esac
