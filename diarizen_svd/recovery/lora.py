#!/usr/bin/env python3
"""Sequential LoRA on the two factors of a low-rank layer (paper §2.6).

Every compressed matrix is already a factor pair  W ~= W'_u W'_v  (LowRankLinear.lr_A / lr_B).
Following the sequential low-rank update of SVD-LLM [Wang et al., ICLR 2025;
github.com/AIoT-MLSys-Lab/SVD-LLM, "SVD-LLM (LoRA fine-tuning)"] the two factors are trained one
after the other with LoRA adapters [Hu et al., ICLR 2022]:

    Phase A:  W'_u <- W'_u + B_u A_u   (W'_v frozen)      then merge
    Phase B:  W'_v <- W'_v + B_v A_v   (W'_u frozen)      then merge

The adapters are attached to ``LowRankLinear`` only.  The CNN factor pairs
(``LowRankConv1d.conv_A / conv_B``, 4.2M parameters) are small enough to be trained directly and
carry no adapters.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from diarizen_svd.nn.lowrank import LowRankLinear

LORA_KEYS = ("upA", "downA", "upB", "downB")


def inject_lora(model, rank):
    """Attach zero-initialised LoRA adapters to both factors of every LowRankLinear (up = 0, down ~ N(0, 0.01))."""
    mods = []
    for m in model.modules():
        if isinstance(m, LowRankLinear):
            r, do, di = m.rank, m.out_features, m.in_features
            dev = m.lr_A.weight.device
            m.upA = nn.Parameter(torch.zeros(do, rank, device=dev))
            m.downA = nn.Parameter(0.01 * torch.randn(rank, r, device=dev))
            m.upB = nn.Parameter(torch.zeros(r, rank, device=dev))
            m.downB = nn.Parameter(0.01 * torch.randn(rank, di, device=dev))
            m.act_A = m.act_B = False

            def fwd(x, m=m):
                Bw = m.lr_B.weight + (m.upB @ m.downB if m.act_B else 0)
                Aw = m.lr_A.weight + (m.upA @ m.downA if m.act_A else 0)
                return F.linear(F.linear(x, Bw, None), Aw, m.lr_A.bias)
            m.forward = fwd
            mods.append(m)
    return mods


def set_phase(mods, phase):
    a, b = phase == "A", phase == "B"
    for m in mods:
        m.act_A, m.act_B = a, b
        m.upA.requires_grad_(a); m.downA.requires_grad_(a)
        m.upB.requires_grad_(b); m.downB.requires_grad_(b)


def merge(mods, phase):
    for m in mods:
        if phase == "A": m.lr_A.weight.data += (m.upA @ m.downA).to(m.lr_A.weight.dtype)
        if phase == "B": m.lr_B.weight.data += (m.upB @ m.downB).to(m.lr_B.weight.dtype)


def save_effective(model, mods, path):
    """Temporarily merge the active adapters, save a plain state_dict (no LoRA keys), un-merge."""
    for m in mods:
        if m.act_A: m.lr_A.weight.data += (m.upA @ m.downA)
        if m.act_B: m.lr_B.weight.data += (m.upB @ m.downB)
    sd = {k: v for k, v in model.state_dict().items() if not any(t in k for t in LORA_KEYS)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(sd, path)
    for m in mods:
        if m.act_A: m.lr_A.weight.data -= (m.upA @ m.downA)
        if m.act_B: m.lr_B.weight.data -= (m.upB @ m.downB)
