#!/usr/bin/env python3
"""Parameters and MACs of WavLM encoders, split into CNN front-end and Transformer.

Measurement lives in :func:`diarizen_svd.bench.profile`; this is the command-line front end.
A model is given as  name=<path.bin>  or  name=config:<name>  (see diarizen_svd/bench.py).

Usage:
  python scripts/bench_macs.py --models "dense=pretrained/wavlm-large-ft-compound7.bin" \
      "pruned80=config:wavlm_large_s80_md" "lowrank5x=exp/stage1/obdllm-r5/wavlm-large-ft-obdllm-r5.bin"
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarizen_svd.bench import build, profile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=<path.bin> or name=config:<config_name>")
    ap.add_argument("--secs", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = []
    for spec in args.models:
        name, src = spec.split("=", 1)
        r = dict(model=name, source=src, **profile(build(src), args.secs))
        rows.append(r)
        print(f"[macs] {name:<14} params {r['params_M']:7.2f}M (cnn {r['params_cnn_M']:5.2f}M, trans {r['params_transformer_M']:7.2f}M) | "
              f"MACs {r['macs_G']:6.3f}G (cnn {r['macs_cnn_G']:5.3f}G, trans {r['macs_transformer_G']:6.3f}G)", flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
