#!/usr/bin/env python3
"""Calibration statistics for activation- and curvature-aware factorization.

Two statistics are collected on the compound training set, balanced over the seven corpora:

  C_x = sum_t x_t x_t^T        input second moment of every compressed matrix (forward hooks).
                               This is the "truncation-aware data whitening" statistic of
                               SVD-LLM [Wang et al., ICLR 2025; github.com/AIoT-MLSys-Lab/SVD-LLM].
                               For a Conv1d the input is unfolded into im2col patches
                               [Chellapilla et al., IWFHR 2006] so that the convolution becomes one
                               matrix product, as in BALF [Gonzalez et al., arXiv:2509.25136].
  C_g = (1/N) sum_t g_t g_t^T  covariance of the loss gradient w.r.t. the matrix OUTPUT z = W x
                               (full backward hooks).  Together with C_x this is the Kronecker-
                               factored (K-FAC) Hessian H ~= C_x (x) C_g [Martens & Grosse, ICML 2015;
                               Grosse & Martens, ICML 2016 for conv], the curvature model used by
                               OBD-LLM [Li et al., arXiv:2604.00821] and GFWSVD [Chekalina et al.,
                               arXiv:2505.17974].
  I_row = sum_b sum_j (dL/dW_ij)^2   row sums of the empirical weight Fisher, for the FWSVD baseline
                               [Hsu et al., ICLR 2022] "as published" (Eqs. 3 and 6 of that paper).
                               Note: FWSVD accumulates per-sample squared gradients; like the
                               reference implementations we accumulate per mini-batch.

The loss for the backward statistics is the diarization objective itself (powerset PIT-NLL of the
fine-tuned DiariZen model), evaluated in eval mode (dropout off).
"""
import time
import torch

from diarizen_svd.paths import set_seed
from diarizen_svd.data.corpus import CORPORA, per_corpus_lists
from diarizen_svd.data.audio import load_wav
from diarizen_svd.data.loader import make_loader, write_corpus_subset
from diarizen_svd.experiment import diar_loss
from diarizen_svd.compress.targets import CONV_IDX, target_modules


@torch.no_grad()
def collect_input_stats(wavlm, scp, win_per_corpus, conv_idx=CONV_IDX, device="cuda", seed=0):
    """Forward-only pass over ``win_per_corpus`` 16 s windows of every corpus.

    Returns (cov_x: {key: (d_in, d_in) float32 tensor on ``device``}, windows-per-corpus, seconds).
    ``cov_x`` is an unnormalized sum; the factorization damps it relative to its mean diagonal, so the
    scale is irrelevant.  Conv layers use the im2col patch (C_i * K) as the input vector.
    """
    wavlm.eval().to(device)
    mods = target_modules(wavlm, conv_idx)
    cov, hooks = {}, []

    def acc(key, x):
        c = x.t() @ x
        cov[key] = c if key not in cov else cov[key] + c

    def hook_lin(key):
        return lambda m, inp: acc(key, inp[0].detach().reshape(-1, inp[0].shape[-1]).float())

    def hook_conv(key, k, s):
        def h(m, inp):                                   # (B, C_i, T) -> im2col patches (B*T', C_i*K)
            x = inp[0].detach()
            p = x.unfold(2, k, s).permute(0, 2, 1, 3).reshape(-1, x.shape[1] * k).float()
            acc(key, p)
        return h

    for key, m in mods.items():
        if key[0] == "lin":
            hooks.append(m.register_forward_pre_hook(hook_lin(key)))
        else:
            hooks.append(m.register_forward_pre_hook(hook_conv(key, m.kernel_size[0], m.stride[0])))

    wins, per = [], {}
    for corp, recs in per_corpus_lists(scp, seed).items():
        if corp not in CORPORA:
            continue
        cw = []
        for _, path in recs:
            try:
                cw += load_wav(path)
            except Exception as e:
                print(f"[calib] skip {path}: {e}", flush=True)
            if len(cw) >= win_per_corpus:
                break
        cw = cw[:win_per_corpus]
        wins += cw
        per[corp] = len(cw)
    print(f"[calib] windows per corpus: {per}  total={len(wins)}", flush=True)
    t0 = time.perf_counter()
    for w in wins:
        wavlm.extract_features(w.unsqueeze(0).to(device))
    secs = time.perf_counter() - t0
    for h in hooks:
        h.remove()
    return cov, per, secs


def collect_backward_stats(model, scp, rttm, uem, chunk_per_corpus, batch, tmp_prefix, conv_idx=CONV_IDX,
                           want=("cov_g",), seed=0):
    """One forward + backward sweep of the diarization loss over ``chunk_per_corpus`` 8 s chunks per corpus.

    ``model``  : DiariZen ``Model`` (dense WavLM + trained head), moved to CUDA here.
    ``want``   : any of "cov_g" (output-gradient covariance, per matrix) and "row_fisher" (row sums of
                 the squared weight gradient, per matrix).  Both are accumulated in the same sweep.
    Returns ({name: {key: tensor}}, chunks-per-corpus, seconds).
    """
    set_seed(seed)                                       # also seeds the DataLoader shuffling
    model.cuda().eval()                                  # eval = dropout off; grads still flow
    for p in model.parameters():
        p.requires_grad_(True)
    mods = target_modules(model.wavlm_model, conv_idx)
    G, N, I, hooks = {}, {}, {}, []

    def acc(key, g):                                     # g: (frames, d_out) fp32
        G[key] = g.T @ g if key not in G else G[key] + g.T @ g
        N[key] = N.get(key, 0) + g.shape[0]

    def hook_lin(key):
        return lambda m, gi, go: acc(key, go[0].detach().reshape(-1, go[0].shape[-1]).float())

    def hook_conv(key):
        return lambda m, gi, go: acc(key, go[0].detach().permute(0, 2, 1).reshape(-1, go[0].shape[1]).float())

    if "cov_g" in want:
        for key, m in mods.items():
            hooks.append(m.register_full_backward_hook(hook_lin(key) if key[0] == "lin" else hook_conv(key)))

    rttm_lines, uem_lines = open(rttm).readlines(), open(uem).readlines()
    per, t0 = {}, time.perf_counter()
    for corp, recs in per_corpus_lists(scp, seed).items():
        if corp not in CORPORA:
            continue
        s, r, u = write_corpus_subset(tmp_prefix, corp, recs, rttm_lines, uem_lines)
        nc = 0
        for b in make_loader(model, s, r, u, batch, shuffle=True):
            xs, target = b["xs"].cuda(), b["ts"].cuda()
            loss = diar_loss(model, xs, target)
            model.zero_grad(set_to_none=True)
            loss.backward()
            if "row_fisher" in want:
                with torch.no_grad():
                    for key, m in mods.items():
                        rs = (m.weight.grad.float() ** 2).reshape(m.weight.shape[0], -1).sum(1)
                        I[key] = rs.clone() if key not in I else I[key] + rs
            nc += xs.size(0)
            if nc >= chunk_per_corpus:
                break
        per[corp] = nc
        print(f"[backward] {corp}: {nc} chunks", flush=True)
    secs = time.perf_counter() - t0
    for h in hooks:
        h.remove()
    out = {}
    if "cov_g" in want:
        out["cov_g"] = {k: G[k] / max(1, N[k]) for k in G}
    if "row_fisher" in want:
        out["row_fisher"] = I
    model.zero_grad(set_to_none=True)
    return out, per, secs
