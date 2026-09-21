#!/usr/bin/env python3
"""Extract the diarization-fine-tuned WavLM from a DiariZen EEND checkpoint as a standalone
``{config, state_dict}`` WavLM checkpoint (the "teacher" that scripts/build_lowrank.py compresses).

The fine-tuned encoder has the architecture of the converted pre-trained checkpoint, so its config
is borrowed from there (``convert_wavlm_from_hf.py`` of ``recipes/diar_ssl_pruning`` produces it);
only the architecture is taken, never the weights.  This is the same step as
``get_wavlm_from_finetuned.py`` in the pruning recipe of Han et al.

Usage:
  python scripts/extract_ft_wavlm.py --diar_ckpt pretrained/eend_wavlm_large_compound7_avg5.bin \
         --config_src pretrained/wavlm-large-converted.bin \
         --out pretrained/wavlm-large-ft-compound7.bin
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarizen_svd.paths import require
from diarizen.models.module.wav2vec2.model import wav2vec2_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diar_ckpt", required=True, help="DiariZen EEND checkpoint (state_dict with wavlm_model.* keys)")
    ap.add_argument("--config_src", required=True, help="converted pre-trained WavLM checkpoint (architecture config)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    require(args.diar_ckpt, "DiariZen EEND checkpoint fine-tuned on the compound set",
            "train it with DiariZen recipes/diar_ssl/run_stage.sh, or download the authors' release",
            "--diar_ckpt")
    require(args.config_src, "converted pre-trained WavLM-Large (architecture config only)",
            "DiariZen recipes/diar_ssl_pruning/convert_wavlm_from_hf.py", "--config_src")

    diar = torch.load(args.diar_ckpt, map_location="cpu")
    sd = diar.get("state_dict", diar) if isinstance(diar, dict) else diar
    pref = "wavlm_model."
    wavlm_sd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    head = sorted({k.split(".")[0] for k in sd if not k.startswith(pref)})
    print(f"EEND checkpoint: {len(sd)} tensors | WavLM: {len(wavlm_sd)} | head modules: {head}")

    src = torch.load(args.config_src, map_location="cpu")
    config = dict(src["config"])
    m = wav2vec2_model(**config)
    miss, unexp = m.load_state_dict(wavlm_sd, strict=False)
    print(f"load into converted config: missing={len(miss)} unexpected={len(unexp)}")
    kk = "encoder.transformer.layers.6.attention.q_proj.weight"
    if kk in wavlm_sd and kk in src["state_dict"]:
        d = (wavlm_sd[kk].float() - src["state_dict"][kk].float()).abs().mean().item()
        print(f"mean |fine-tuned - pre-trained| at L6 q_proj = {d:.3e} (> 0: genuinely fine-tuned)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"config": config, "state_dict": wavlm_sd}, args.out)
    P = sum(v.numel() for v in wavlm_sd.values() if torch.is_floating_point(v))
    print(f"saved fine-tuned WavLM ({P/1e6:.2f}M) -> {args.out}")


if __name__ == "__main__":
    main()
