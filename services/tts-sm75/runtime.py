from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from models.breeze import BreezeForConditionalGeneration


def get_dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def resolve_device(explicit_device: str | None = None) -> str:
    if explicit_device:
        return explicit_device

    _, _, local_rank = get_dist_info()
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_generation_config_for_breeze(
    model: torch.nn.Module,
    generation_config: dict[str, Any] | None = None,
) -> None:
    generation_config = generation_config or {
        "depth_decoder_do_sample": True,
        "depth_decoder_temperature": 0.9,
        "depth_decoder_top_p": 1.0,
        "depth_decoder_top_k": 50,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "max_new_tokens": 750,
        "temperature": 0.9,
    }

    prefix = "depth_decoder_"
    depth_decoder_attrs = {
        attr[len(prefix) :]: value
        for attr, value in generation_config.items()
        if attr.startswith(prefix)
    }
    vars(model.depth_decoder.generation_config).update(
        {"_from_model_config": False, **depth_decoder_attrs}
    )
    vars(model.generation_config).update(generation_config)


def _keep_text_encoder_fp32(model, dtype) -> None:
    """Rozmova, fp16 on Turing: the T5Gemma2 text encoder's residual stream exceeds fp16's range (inf, then NaN in the
    sampler). Run it and its projections in fp32 and hand the projected states to the fp16 backbone."""
    import torch

    model.text_encoder.float()
    for name in ("text_encoder_proj", "text_encoder_layer_projs"):
        if getattr(model, name, None) is not None:
            getattr(model, name).float()
    project = model._project_segments
    logged = []

    fp16_max = torch.finfo(dtype).max

    def _narrow(x):
        """fp32 -> fp16 for the backbone; values beyond fp16's range are clamped (warned once) instead of becoming
        inf, which would poison the CUDA context through the sampler's device-side assert."""
        peak = float(x.abs().max())
        if not torch.isfinite(x).all() or peak > fp16_max:
            if "clamp" not in logged:
                print(f"rozmova: WARNING text encoder output |x| max {peak} exceeds fp16; clamping", flush=True)
                logged.append("clamp")
            x = torch.nan_to_num(x, nan=0.0).clamp(-fp16_max, fp16_max)
        return x.to(dtype)

    def _project_segments(*args, **kwargs):
        text_embeds, layer_states = project(*args, **kwargs)
        if not logged:
            peak = max([float(text_embeds.abs().max())] + [float(x.abs().max()) for x in layer_states or []])
            print(f"rozmova: text encoder in fp32, projected |x| max {peak:.1f} (fp16 max 65504)", flush=True)
            logged.append(peak)
        return _narrow(text_embeds), [_narrow(x) for x in layer_states] if layer_states is not None else None

    model._project_segments = _project_segments


def load_runtime(
    ckpt_dir: Path,
    *,
    device: str,
    attn_implementation: str,
) -> tuple[AutoTokenizer, BreezeForConditionalGeneration, Any]:

    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            rank, world_size, local_rank = get_dist_info()
            raise RuntimeError(
                "Failed to set CUDA device "
                f"device={device} rank={rank} world_size={world_size} local_rank={local_rank} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"device_count={torch.cuda.device_count()}"
            ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        fix_mistral_regex=False,
    )
    model = BreezeForConditionalGeneration.from_pretrained(
        ckpt_dir,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    # Rozmova: BREEZE_DTYPE=float16 on Turing (no fast bf16 on sm_75), with the fp32 mask fix in models/breeze.py.
    # Cast the whole model, since the text encoder is built in bf16 explicitly (models/breeze.py). Unset = bf16.
    _dtype = os.environ.get("BREEZE_DTYPE")
    model.to(device=device, dtype=getattr(torch, _dtype)).eval() if _dtype else model.to(device).eval()
    if _dtype == "float16" and getattr(model, "text_encoder", None) is not None:
        _keep_text_encoder_fp32(model, torch.float16)

    from qwen_tts import Qwen3TTSTokenizer

    bundled_audio_tokenizer = ckpt_dir / "audio_tokenizer"
    if not bundled_audio_tokenizer.is_dir():
        raise FileNotFoundError(
            "Bundled audio tokenizer not found at "
            f"{bundled_audio_tokenizer}. The Breeze model package must include "
            "the audio_tokenizer directory."
        )
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(bundled_audio_tokenizer), device_map=os.environ.get("BREEZE_TOKENIZER_DEVICE", device)
    )
    return tokenizer, model, audio_tokenizer
