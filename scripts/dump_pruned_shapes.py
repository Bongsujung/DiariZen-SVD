#!/usr/bin/env python3
"""Record the per-matrix shapes of a structured-pruned WavLM (Han et al., TASLP 2026) as JSON.

Structured pruning removes whole attention heads and whole feed-forward units, so Q/K/V/Out of a
layer share one width and FFN-in/FFN-out share another; a fully pruned attention block leaves no
weight at all (the key is then absent).  The dump is the per-matrix counterpart of a low-rank build's
``build_dump/per_matrix.tsv``, for comparing the two compression patterns on the same grid.

Usage: python scripts/dump_pruned_shapes.py <pruned_state_dict.bin> <out.json>
"""
import argparse
import json
import torch

LEAVES = ("q_proj", "k_proj", "v_proj", "out_proj", "intermediate_dense", "output_dense")


def pruned_shapes(sd):
    """{'conv': {idx: shape}, 'lin': {layer: {leaf: shape}}} from a (possibly EEND-prefixed) state_dict."""
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
    pref = "wavlm_model." if any(k.startswith("wavlm_model.") for k in sd) else ""
    out = {"conv": {}, "lin": {}}
    for k, v in sd.items():
        if not k.startswith(pref):
            continue
        k = k[len(pref):]
        if k.startswith("feature_extractor.conv_layers.") and k.endswith(".conv.weight"):
            out["conv"][k.split(".")[2]] = list(v.shape)
        elif k.startswith("encoder.transformer.layers.") and k.endswith(".weight") and v.dim() == 2:
            parts = k.split(".")
            layer, leaf = parts[3], parts[-2]
            if leaf in LEAVES:
                out["lin"].setdefault(layer, {})[leaf] = list(v.shape)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="pruned checkpoint (EEND state_dict or bare WavLM state_dict)")
    ap.add_argument("dst", help="output .json")
    args = ap.parse_args()
    out = pruned_shapes(torch.load(args.src, map_location="cpu"))
    json.dump(out, open(args.dst, "w"))
    print(f"conv layers {len(out['conv'])}, transformer layers {len(out['lin'])} -> {args.dst}")


if __name__ == "__main__":
    main()
