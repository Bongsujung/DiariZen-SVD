#!/usr/bin/env python3
"""Equal-norm re-split of every low-rank factor pair in a compressed WavLM checkpoint (paper Eq. 7).

For each pair (A, B) with W ~= A B  (LowRankLinear.lr_A / lr_B and LowRankConv1d.conv_A / conv_B):
    s = sqrt(||A||_F / ||B||_F),   A <- A / s,   B <- B * s.
The product A B -- and therefore the compressed model -- is unchanged; only the split of the scale
between the two factors is, which is what conditions the recovery fine-tuning of scripts/recover.py.
The convention is that of PiSSA [Meng et al., NeurIPS 2024] and RefLoRA [Zhang et al., NeurIPS 2025].

scripts/build_lowrank.py applies this by default; this script exists to re-split a checkpoint that
was built with --no_resplit (or by older code) without recomputing anything.
"""
import os
import sys
import argparse
import statistics
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarizen_svd.compress.factorize import resplit_state_dict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="input checkpoint ({config, state_dict} or a bare state_dict)")
    ap.add_argument("dst", help="output checkpoint")
    args = ap.parse_args()
    ck = torch.load(args.src, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd, before, after, maxrel = resplit_state_dict(sd)
    print(f"[resplit] {len(before)} factor pairs")
    print(f"[resplit] ||A||/||B|| before: median {statistics.median(before):.3e}  max {max(before):.3e}")
    print(f"[resplit] ||A||/||B|| after : median {statistics.median(after):.3e}  max {max(after):.3e}")
    print(f"[resplit] max relative change of ||A|| ||B||: {maxrel:.3e}")
    torch.save(ck if isinstance(ck, dict) and "state_dict" in ck else sd, args.dst)
    print(f"[resplit] saved {args.dst}")


if __name__ == "__main__":
    main()
