#!/usr/bin/env python3
"""Repository paths, default asset locations and seeding.

Every script and library module resolves its defaults through :data:`DEFAULT`; each one is also
overridable on the command line, so nothing here is load-bearing for a user with a different layout.
"""
import os
import random
import sys

import numpy as np
import torch

# ----------------------------------------------------------------------------- paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))      # repository root
RECIPE = ROOT
# The repository root must be importable: ``data/loader.py`` reaches for the top-level ``dataset.py``
# (a verbatim copy of DiariZen ``recipes/diar_ssl/dataset.py``) with a plain ``from dataset import ...``.
# A ``pyproject.toml`` that installs the package would let this go away.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT = dict(
    # dense, diarization-fine-tuned WavLM-Large ({config, state_dict}); written by stage 0
    teacher_wavlm=f"{ROOT}/exp/stage0/wavlm-large-ft.bin",
    # the matching DiariZen EEND checkpoint (WavLM + weighted sum + Conformer head), avg of 5 epochs
    teacher_diar=f"{ROOT}/pretrained/eend_wavlm_large_compound7_avg5.bin",
    # calibration: the seven training splits pooled (data/<corpus>/train -> data/_pooled/train, scripts/prepare_data.py)
    train_scp=f"{ROOT}/data/_pooled/train/wav.scp",
    train_rttm=f"{ROOT}/data/_pooled/train/rttm",
    train_uem=f"{ROOT}/data/_pooled/train/all.uem",
    # recovery: 50 % of every training split, and the pooled dev splits
    recovery_train=f"{ROOT}/data/_pooled/recover",
    recovery_dev=f"{ROOT}/data/_pooled/dev",
    # where compressed encoders are written
    lowrank_dir=f"{ROOT}/exp/stage1",
)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def require(path: str, what: str, how: str, flag: str = ""):
    """Fail early with an actionable message instead of a ``FileNotFoundError`` deep in torch.load."""
    if os.path.exists(path):
        return path
    raise SystemExit(
        f"\n[missing asset] {what}\n"
        f"  expected at : {path}\n"
        f"  how to get  : {how}\n"
        + (f"  or point at an existing copy with {flag}\n" if flag else "")
        + "  see conf/run.toml and README.md\n")


def require_build_assets(args):
    """Preflight for the compression pipeline: teacher encoder, EEND head, calibration lists."""
    require(args.teacher_wavlm, "dense fine-tuned WavLM-Large encoder (the compression teacher)",
            "bash run_stage.sh 0   (extracts it from the EEND checkpoint)", "--teacher_wavlm")
    require(args.teacher_diar, "DiariZen EEND checkpoint (supplies the trained head and the calibration loss)",
            "train it with DiariZen recipes/diar_ssl/run_stage.sh, or download the authors' release",
            "--teacher_diar")
    for p, flag in ((args.train_scp, "--train_scp"), (args.train_rttm, "--train_rttm"), (args.train_uem, "--train_uem")):
        require(p, "calibration data list", "see data/README.md for the expected layout", flag)
