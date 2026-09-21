#!/usr/bin/env python3
"""Average the epoch checkpoints of a recovery run and assemble an experiment directory for eval_tf.sh.

Averaging lives in :func:`diarizen_svd.recovery.average.average_state_dicts`; this is the
command-line front end.  The paper averages the five Phase-B epoch checkpoints written by
``scripts/recover.py``.

Usage:
  python scripts/average_epochs.py --recover_exp exp/stage2/obdllm-r5_recover --out_exp exp/stage2/obdllm-r5_recover_avg5 \
         --wavlm_bin exp/stage1/obdllm-r5/wavlm-large-ft-obdllm-r5.bin [--ckpts B_ep01,B_ep02,B_ep03,B_ep04,B_ep05]
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarizen_svd.experiment import HEAD_LARGE, write_exp_scaffold
from diarizen_svd.recovery.average import average_state_dicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recover_exp", required=True, help="recover.py experiment directory (has ep_ckpts/)")
    ap.add_argument("--out_exp", required=True)
    ap.add_argument("--wavlm_bin", required=True, help="the compressed WavLM the run started from (for the config)")
    ap.add_argument("--ckpts", default="B_ep01,B_ep02,B_ep03,B_ep04,B_ep05")
    args = ap.parse_args()
    paths = [os.path.join(args.recover_exp, "ep_ckpts", f"{c}.bin") for c in args.ckpts.split(",")]
    for p in paths:
        assert os.path.exists(p), f"missing {p}"
    avg = average_state_dicts(paths)
    torch.save(avg, write_exp_scaffold(args.out_exp, args.wavlm_bin, HEAD_LARGE))
    print(f"[average] {len(paths)} checkpoints -> {args.out_exp} ({len(avg)} tensors)")


if __name__ == "__main__":
    main()
