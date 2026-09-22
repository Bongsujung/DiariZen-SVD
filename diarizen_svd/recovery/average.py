#!/usr/bin/env python3
"""Checkpoint averaging over a recovery trajectory.

The paper averages the five Phase-B epoch checkpoints written by ``scripts/recover.py``.

"""
import torch


def average_state_dicts(paths):
    sds = [torch.load(p, map_location="cpu") for p in paths]
    avg = {}
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            avg[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype)
        else:
            avg[k] = sds[-1][k].clone()
    return avg
