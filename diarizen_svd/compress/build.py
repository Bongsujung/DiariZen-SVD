#!/usr/bin/env python3
"""Building a low-rank encoder out of a dense one: statistics, factorization, caching, construction.

The four steps of the training-free pipeline (paper §2) live here; ``scripts/build_lowrank.py`` is the
command-line front end that wires them together, allocates the budget and writes the checkpoint.

  1. calibration      C_x from 150 x 16 s windows per corpus (forward); C_g and/or the row Fisher from
                      600 x 8 s chunks per corpus of the diarization loss (one backward)   [calibrate.py]
  2. factorization    one of  svdllm | fwsvd | obdllm   for all 144 linear + 6 conv matrices [factorize.py]
  3. allocation       one global parameter budget P_full / rho, greedy water-filling        [allocate.py]
  4. construction     LowRankLinear / LowRankConv1d factors, equal-norm re-split, checkpoint

Neither the calibration statistics nor the factorization depend on rho: with ``--stats_cache`` and
``--spectra_cache`` they are computed once, and every further compression ratio only re-runs the
allocation and construction (seconds).  The spectra cache holds U, S, Vh, L_x of all 150 matrices
(~4.0 GB per factorization rule).

The ``args`` argument of :func:`get_stats` / :func:`get_spectra` / :func:`factorize_all` is the
``argparse.Namespace`` of ``scripts/build_lowrank.py``; it is passed through unchanged so that the
cache-validity check sees exactly the settings the user asked for.
"""
import os
import time

import numpy as np
import torch

from diarizen.models.module.wav2vec2.model import wav2vec2_model

from diarizen_svd.experiment import HEAD_LARGE, build_diar_model
from diarizen_svd.nn.lowrank import apply_lowrank
from diarizen_svd.compress.targets import TARGETS, target_modules, key_name, sort_keys
from diarizen_svd.compress.calibrate import collect_input_stats, collect_backward_stats
from diarizen_svd.compress.factorize import (Spectrum, input_whitened_svd, two_sided_svd,
                                             fisher_weighted_svd, to_factors, equal_norm_resplit)
from diarizen_svd.compress.allocate import rank_cost, rank_cap, dense_params

METHODS = {"svdllm": "input-side whitening  SVD(W L_x)            [Wang et al., ICLR 2025]",
           "fwsvd":  "row-Fisher weighting  SVD(D W), no L_x      [Hsu et al., ICLR 2022]",
           "obdllm": "two-sided whitening   SVD(L_g^T W L_x)      [Li et al., 2026; Friedland & Torokhti, 2007]"}


def load_dense(path):
    ck = torch.load(path, map_location="cpu")
    cfg, sd = dict(ck["config"]), ck["state_dict"]
    for k, v in cfg.items():
        if "prune" in k and v is not False:
            raise ValueError(f"the dense checkpoint must be unpruned; found {k}={v}")
    m = wav2vec2_model(**cfg)
    res = m.load_state_dict(sd, strict=False)
    assert not res.missing_keys, f"missing keys in dense checkpoint: {res.missing_keys[:5]}"
    P_full = sum(v.numel() for v in sd.values() if torch.is_floating_point(v))
    return m, cfg, P_full


def get_stats(args, dense, need):
    """Return {cov_x, [cov_g], [row_fisher]} (+ timing), computing what the cache does not hold."""
    stats, timing, per = {}, {}, {}
    if args.stats_cache and os.path.exists(args.stats_cache):
        c = torch.load(args.stats_cache, map_location="cpu")
        stats, timing, per = c["stats"], c.get("timing", {}), c.get("per_corpus", {})
        print(f"[stats] loaded {sorted(stats)} from {args.stats_cache}", flush=True)
    changed = False
    if "cov_x" not in stats:
        cov, per["input_windows"], timing["C_x_forward"] = collect_input_stats(
            dense, args.train_scp, args.win_per_corpus, args.conv_idx, args.device, args.seed)
        stats["cov_x"] = {k: v.cpu() for k, v in cov.items()}
        changed = True
    want = tuple(n for n in need if n in ("cov_g", "row_fisher") and n not in stats)
    if want:
        diar = build_diar_model(args.teacher_wavlm, args.teacher_diar, HEAD_LARGE)
        os.makedirs(os.path.dirname(args.tmp_prefix), exist_ok=True)
        out, per["backward_chunks"], secs = collect_backward_stats(
            diar, args.train_scp, args.train_rttm, args.train_uem, args.chunk_per_corpus, args.batch,
            args.tmp_prefix, args.conv_idx, want=want, seed=args.seed)
        for n in want:
            stats[n] = {k: v.cpu() for k, v in out[n].items()}
            timing[f"{n}_backward"] = secs
        del diar
        torch.cuda.empty_cache()
        changed = True
    if args.stats_cache and changed:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_cache)), exist_ok=True)
        torch.save(dict(stats=stats, timing=timing, per_corpus=per,
                        settings=dict(win_per_corpus=args.win_per_corpus, chunk_per_corpus=args.chunk_per_corpus,
                                      batch=args.batch, conv_idx=args.conv_idx, seed=args.seed,
                                      teacher_wavlm=args.teacher_wavlm, teacher_diar=args.teacher_diar)),
                   args.stats_cache)
        print(f"[stats] saved {sorted(stats)} -> {args.stats_cache}", flush=True)
    return stats, timing, per


def factorize_all(args, dense, stats):
    mods = target_modules(dense, args.conv_idx)
    spectra, ridge = {}, {}
    dev = args.device
    for key, m in mods.items():
        W = m.weight.data.to(dev)
        Cx = stats["cov_x"][key].to(dev)
        if args.method == "svdllm":
            spectra[key] = input_whitened_svd(W, Cx, args.damp)
        elif args.method == "obdllm":
            spectra[key], ridge[key_name(key)] = two_sided_svd(W, Cx, stats["cov_g"][key].to(dev), args.damp)
        else:
            spectra[key] = fisher_weighted_svd(W, stats["row_fisher"][key].to(dev), args.fisher_clamp)
    return spectra, ridge


def get_spectra(args, dense, stats):
    """Factorize all matrices, or load the factorization of an earlier run with identical settings."""
    if args.spectra_cache and os.path.exists(args.spectra_cache):
        c = torch.load(args.spectra_cache, map_location="cpu")
        same = all(c["settings"][k] == v for k, v in _spectra_settings(args).items())
        if same:
            spectra = {k: Spectrum(*[t.to(args.device) if torch.is_tensor(t) else t for t in v]) for k, v in c["spectra"].items()}
            print(f"[spectra] loaded {len(spectra)} matrices from {args.spectra_cache}", flush=True)
            return spectra, c["ridge"], 0.0
        print(f"[spectra] cache settings differ ({c['settings']}); recomputing", flush=True)
    t0 = time.perf_counter()
    spectra, ridge = factorize_all(args, dense, stats)
    secs = time.perf_counter() - t0
    if args.spectra_cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.spectra_cache)), exist_ok=True)
        torch.save(dict(settings=_spectra_settings(args), ridge=ridge,
                        spectra={k: tuple(t.cpu() if torch.is_tensor(t) else t for t in sp) for k, sp in spectra.items()}),
                   args.spectra_cache)
        print(f"[spectra] saved -> {args.spectra_cache}", flush=True)
    return spectra, ridge, secs


def _spectra_settings(args):
    return dict(method=args.method, damp=args.damp, fisher_clamp=args.fisher_clamp, conv_idx=list(args.conv_idx),
                stats_cache=os.path.abspath(args.stats_cache) if args.stats_cache else "",
                teacher_wavlm=os.path.abspath(args.teacher_wavlm), win_per_corpus=args.win_per_corpus,
                chunk_per_corpus=args.chunk_per_corpus, seed=args.seed)


def build_lowrank_model(dense, cfg, spectra, ranks, conv_idx, resplit=True):
    """Instantiate the low-rank WavLM, copy the untouched weights and fill in the factor pairs."""
    n = len(dense.encoder.transformer.layers)
    per_layer = [{t: int(ranks[("lin", li, t)]) for t in TARGETS} for li in range(n)]
    cr = {i: int(ranks[("conv", i)]) for i in conv_idx}
    lr = wav2vec2_model(**cfg)                                   # dense architecture ...
    lr.load_state_dict(dense.state_dict(), strict=True)          # ... with the dense weights (norms, biases, conv0, ...)
    apply_lowrank(lr, per_layer, cr)                             # ... then the target layers become factor pairs
    cfg_lr = dict(cfg, lowrank_ranks=per_layer, cnn_lowrank_ranks=cr)
    dense_mods, lr_mods = target_modules(dense, conv_idx), target_modules(lr, conv_idx)
    scales = {}
    for key, sp in spectra.items():
        A, B = to_factors(sp, int(ranks[key]))
        if resplit:
            A, B, scales[key_name(key)] = equal_norm_resplit(A, B)
        m, base = lr_mods[key], dense_mods[key]
        if key[0] == "lin":
            m.lr_A.weight.data.copy_(A.to(m.lr_A.weight.dtype))
            m.lr_B.weight.data.copy_(B.to(m.lr_B.weight.dtype))
            if base.bias is not None:
                m.lr_A.bias.data.copy_(base.bias.data.to(m.lr_A.bias.dtype))
        else:
            o, ic, k = sp.shape
            r = int(ranks[key])
            m.conv_A.weight.data.copy_(A.reshape(o, r, 1).to(m.conv_A.weight.dtype))
            m.conv_B.weight.data.copy_(B.reshape(r, ic, k).to(m.conv_B.weight.dtype))
            if base.bias is not None:
                m.conv_A.bias.data.copy_(base.bias.data.to(m.conv_A.bias.dtype))
    P = sum(p.numel() for p in lr.parameters())
    return {"config": cfg_lr, "state_dict": lr.cpu().state_dict()}, P, scales


def dump_allocation(dump_dir, spectra, ranks):
    os.makedirs(dump_dir, exist_ok=True)
    cols = ["key", "type", "layer", "dtype", "dout", "din", "cap", "rank", "cost", "comp_rate", "sigma2_sum", "sigma2_kept_frac"]
    npz = {}
    with open(os.path.join(dump_dir, "per_matrix.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for key in sort_keys(spectra):
            sp, r = spectra[key], int(ranks[key])
            shp = sp.shape
            dout, din = (shp[0], shp[1]) if len(shp) == 2 else (shp[0], shp[1] * shp[2])
            dtype = key[2].split(".")[-1] if key[0] == "lin" else "conv"
            s2 = (sp.S.detach().cpu().double() ** 2).numpy()
            f.write("\t".join(str(x) for x in [key_name(key), key[0], key[1], dtype, dout, din,
                                                rank_cap(shp, sp.S.numel()), r, rank_cost(shp),
                                                round(1 - r * rank_cost(shp) / dense_params(shp), 4),
                                                f"{s2.sum():.4e}", round(float(s2[:r].sum() / s2.sum()), 4)]) + "\n")
            npz[key_name(key).replace(".", "_") + "_sigma2"] = s2
    np.savez_compressed(os.path.join(dump_dir, "sigma2_per_component.npz"), **npz)
