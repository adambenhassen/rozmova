# Patched Triton unified attention vs a PyTorch reference, fp16, on this GPU. Fails on NaN or error > 5e-3:
#   docker run --rm --gpus device=0 --entrypoint python3 \
#     -v $PWD/services/llm-sm75/triton_unified_attention.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py:ro \
#     -v $PWD/services/llm-sm75/test_attention.py:/t.py:ro docker.io/vllm/vllm-openai:v0.26.0 /t.py
import math, sys, torch, torch.nn.functional as F
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

torch.manual_seed(0)
dev, dt, TOL = "cuda", torch.float16, 5e-3


def run(D, ctx, qlen, block=16, hq=8, hkv=2, window=-1, use_3d=False, segments=16):
    total = ctx + qlen
    nb = math.ceil(total / block) + 1
    kc = torch.randn(nb, block, hkv, D, device=dev, dtype=dt); vc = torch.randn_like(kc)
    q = torch.randn(qlen, hq, D, device=dev, dtype=dt)
    out = torch.full_like(q, float("nan"))                      # poisoned: every output element must be written
    extra = {}
    if use_3d:                                                  # segmented softmax + reduction (decode path)
        hp = 1 << (D - 1).bit_length()
        extra = dict(seq_threshold_3D=1 << 30, num_par_softmax_segments=segments,
                     softmax_segm_output=torch.full((qlen, hq, segments, hp), float("nan"), device=dev),
                     softmax_segm_max=torch.full((qlen, hq, segments), float("nan"), device=dev),
                     softmax_segm_expsum=torch.full((qlen, hq, segments), float("nan"), device=dev))
    unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=torch.tensor([0, qlen], device=dev, dtype=torch.int32),
                      max_seqlen_q=qlen, seqused_k=torch.tensor([total], device=dev, dtype=torch.int32), max_seqlen_k=total,
                      softmax_scale=1 / math.sqrt(D), causal=True, window_size=(window - 1, 0) if window > 0 else (-1, -1),
                      block_table=torch.arange(nb, device=dev, dtype=torch.int32)[None], softcap=0,
                      q_descale=None, k_descale=None, v_descale=None, **extra)
    k = kc.reshape(-1, hkv, D)[:total].repeat_interleave(hq // hkv, 1)
    v = vc.reshape(-1, hkv, D)[:total].repeat_interleave(hq // hkv, 1)
    qpos = torch.arange(ctx, total, device=dev)[:, None]; kpos = torch.arange(total, device=dev)[None]
    mask = kpos <= qpos
    if window > 0:
        mask &= kpos > qpos - window
    ref = F.scaled_dot_product_attention(q.transpose(0, 1).float(), k.transpose(0, 1).float(), v.transpose(0, 1).float(),
                                         attn_mask=mask[None]).transpose(0, 1)
    if not torch.isfinite(out).all():
        return float("nan")
    return float((out.float() - ref).abs().max())


cases = [(ctx, qlen, w, blk, d3) for ctx, qlen, w in ((0, 37, -1), (0, 300, -1), (120, 1, -1), (500, 1, -1),
                                                       (0, 300, 128), (700, 1, 128), (333, 1, -1))
         for blk in (16, 32) for d3 in ((False, True) if qlen == 1 else (False,))]
failed = 0
for D in (256, 512):
    for ctx, qlen, w, blk, d3 in cases:
        err = run(D, ctx, qlen, block=blk, window=w, use_3d=d3)
        ok = err == err and err <= TOL
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} D={D} ctx={ctx} qlen={qlen} window={w} block={blk} 3d={d3}: max|err| = {err:.4f}")
print(f"{failed} failed")
sys.exit(1 if failed else 0)
