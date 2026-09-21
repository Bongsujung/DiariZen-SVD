#!/usr/bin/env python3
"""The recovery objective (paper Eq. 8) and the dev-loss used for model selection.

    L = L_diar + lambda * sum_{i in S} sum_t [ L1(h_t^i, h^_t^i) - cos(h_t^i, h^_t^i) ]

``L_diar`` is the powerset PIT-NLL of DiariZen [Plaquet & Bredin, Interspeech 2023]; the layer-wise
L1 + cosine feature-distillation term follows DistilHuBERT [Chang et al., ICASSP 2022] and DPHuBERT
[Peng et al., Interspeech 2023], with the dense fine-tuned WavLM as teacher.  Layer 0 is the CNN
output (input of Transformer layer 0).  The layer set of the paper is S = {0, 2, 4, 6, 8, 16, 24}
(the {0, 8, 16, 24} of Han et al. [TASLP 2026] plus the early layers, where the rank allocation
concentrates its budget).
"""
import torch
import torch.nn.functional as F

from diarizen_svd.experiment import diar_loss


class FeatureTap:
    """Captures the stacked per-layer WavLM features (B, T, D, n_layers) of a DiariZen ``Model`` with a
    forward pre-hook on its layer-weighting module, so one student forward serves both loss terms."""

    def __init__(self, model):
        self.feat = None
        self.h = model.weight_sum.register_forward_pre_hook(lambda m, inp: setattr(self, "feat", inp[0]))


def recovery_loss(model, teacher, xs, target, S, lam, mode, tap):
    """task | distill | both, from ONE student forward (features read through ``tap``)."""
    yp = model(xs)
    loss = 0.0
    if mode in ("task", "both"):
        from pyannote.audio.utils.permutation import permutate
        from pyannote.audio.utils.loss import nll_loss
        ml = model.powerset.to_multilabel(yp)
        pt, _ = permutate(ml, target)
        ptp = model.powerset.to_powerset(pt.float())
        loss = loss + nll_loss(yp, torch.argmax(ptp, dim=-1))
    if mode in ("distill", "both"):
        hs = tap.feat                                             # (B, T, D, n_layers), with grad
        with torch.no_grad():
            ht = teacher.wav2wavlm(xs[:, model.selected_channel, :], teacher.wavlm_model)
        dl = 0.0
        for i in S:
            s, t = hs[..., i], ht[..., i]
            dl = dl + F.l1_loss(s, t) + (1.0 - F.cosine_similarity(s, t, dim=-1).mean())
        loss = loss + lam * dl
    return loss


@torch.no_grad()
def dev_loss(model, loader):
    model.eval()
    tot = n = 0.0
    for b in loader:
        xs, t = b["xs"].cuda(), b["ts"].cuda()
        tot += diar_loss(model, xs, t).item() * xs.size(0)
        n += xs.size(0)
    model.train()
    return tot / n
