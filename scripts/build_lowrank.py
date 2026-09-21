#!/usr/bin/env python3
"""Compress a diarization-fine-tuned WavLM-Large into a low-rank encoder (training-free).

Command-line front end for :mod:`diarizen_svd.compress.build`; the pipeline itself (calibration,
factorization, caching, construction) lives there, the rank allocation in
:mod:`diarizen_svd.compress.allocate`.  Steps (paper §2):

  1. calibration      C_x (forward) and C_g / row Fisher (one backward of the diarization loss)
  2. factorization    one of  svdllm | fwsvd | obdllm   for all 144 linear + 6 conv matrices
  3. allocation       one global parameter budget P_full / rho, greedy water-filling
  4. construction     LowRankLinear / LowRankConv1d factors, equal-norm re-split, checkpoint
  5. assembly         a DiariZen experiment directory with the teacher's EEND head, ready for eval_tf.sh

Usage (main system of the paper, rho = 5):
  python scripts/build_lowrank.py --method obdllm --ratio 5 --tag obdllm-r5 --stats_cache exp/stats_w150_c600.pt
Baselines of Table 3:
  python scripts/build_lowrank.py --method svdllm --ratio 5 --tag svdllm-r5 --stats_cache exp/stats_w150_c600.pt
  python scripts/build_lowrank.py --method fwsvd  --ratio 5 --tag fwsvd-r5  --stats_cache exp/stats_w150_c600.pt
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarizen_svd.paths import RECIPE, DEFAULT, set_seed, require_build_assets
from diarizen_svd.experiment import HEAD_LARGE, build_diar_model, write_exp_scaffold
from diarizen_svd.compress.build import (METHODS, load_dense, get_stats, get_spectra,
                                         build_lowrank_model, dump_allocation)
from diarizen_svd.compress.allocate import water_fill, dense_params, lowrank_params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=list(METHODS), required=True)
    ap.add_argument("--ratio", type=float, required=True, help="target compression ratio rho (P_full / P_lowrank)")
    ap.add_argument("--tag", required=True, help="name of the output checkpoint / experiment directory")
    ap.add_argument("--teacher_wavlm", default=DEFAULT["teacher_wavlm"])
    ap.add_argument("--teacher_diar", default=DEFAULT["teacher_diar"])
    ap.add_argument("--train_scp", default=DEFAULT["train_scp"])
    ap.add_argument("--train_rttm", default=DEFAULT["train_rttm"])
    ap.add_argument("--train_uem", default=DEFAULT["train_uem"])
    ap.add_argument("--win_per_corpus", type=int, default=150, help="16 s windows per corpus for C_x")
    ap.add_argument("--chunk_per_corpus", type=int, default=600, help="8 s chunks per corpus for C_g / row Fisher")
    ap.add_argument("--batch", type=int, default=8, help="batch size of the backward sweep")
    ap.add_argument("--damp", type=float, default=1e-3, help="Cholesky damping, fraction of the mean diagonal")
    ap.add_argument("--fisher_clamp", type=float, default=1e-6, help="FWSVD: clamp rows below this x mean(I_row)")
    ap.add_argument("--conv_idx", default="1,2,3,4,5,6", help="CNN layers to factorize (conv0 kept dense)")
    ap.add_argument("--round_to", type=int, default=8, help="linear-map ranks are multiples of this")
    ap.add_argument("--floor", type=int, default=1, help="rank floor before water-filling (1 = none)")
    ap.add_argument("--no_resplit", action="store_true", help="skip the equal-norm rescaling of Eq. (7) (paper Fig. 3, 'w/o rescaling')")
    ap.add_argument("--stats_cache", default="", help=".pt file holding C_x / C_g / row Fisher; reused across rho")
    ap.add_argument("--spectra_cache", default="", help=".pt file holding the factorization (U, S, Vh, L_x); reused across rho")
    ap.add_argument("--out_bin", default="", help=f"default {DEFAULT['lowrank_dir']}/wavlm-large-ft-<tag>.bin")
    ap.add_argument("--exp_dir", default="", help=f"default {RECIPE}/exp/stage1/<tag>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.conv_idx = [int(x) for x in args.conv_idx.split(",") if x]
    args.out_bin = args.out_bin or f"{DEFAULT['lowrank_dir']}/wavlm-large-ft-{args.tag}.bin"
    args.exp_dir = args.exp_dir or f"{RECIPE}/exp/stage1/{args.tag}"
    args.tmp_prefix = os.path.join(args.exp_dir, "_calib", "subset")
    dump_dir = os.path.join(args.exp_dir, "build_dump")
    os.makedirs(os.path.dirname(args.out_bin), exist_ok=True)
    os.makedirs(args.exp_dir, exist_ok=True)
    require_build_assets(args)               # fail early, with a pointer to the README, if an asset is missing
    set_seed(args.seed)
    print(f"[build {args.tag}] {args.method}: {METHODS[args.method]} | rho={args.ratio} | conv {args.conv_idx} | "
          f"damp {args.damp} | resplit {not args.no_resplit}", flush=True)

    T = {}
    dense, cfg, P_full = load_dense(args.teacher_wavlm)
    dense.to(args.device).eval()
    need = {"svdllm": ("cov_x",), "obdllm": ("cov_x", "cov_g"), "fwsvd": ("cov_x", "row_fisher")}[args.method]
    stats, T_stats, per = get_stats(args, dense, need)
    T.update(T_stats)

    spectra, ridge, T["factorize"] = get_spectra(args, dense, stats)

    t0 = time.perf_counter()
    P_model = sum(p.numel() for p in dense.parameters())
    P_fixed = P_model - sum(dense_params(sp.shape) for sp in spectra.values())
    ranks = water_fill(spectra, P_full, P_fixed, args.ratio, floor=args.floor, round_to=args.round_to)
    T["allocate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ck, P_lr, scales = build_lowrank_model(dense, cfg, spectra, ranks, args.conv_idx, resplit=not args.no_resplit)
    T["construct"] = time.perf_counter() - t0
    torch.save(ck, args.out_bin)

    early = sum(int(ranks[k]) for k in ranks if k[0] == "lin" and k[1] < 12)
    late = sum(int(ranks[k]) for k in ranks if k[0] == "lin" and k[1] >= 12)
    dump_allocation(dump_dir, spectra, ranks)
    meta = dict(tag=args.tag, method=args.method, ratio=args.ratio, P_full=P_full, P_fixed=P_fixed,
                P_lowrank=P_lr, P_factors=lowrank_params(spectra, ranks), ratio_real=round(P_full / P_lr, 3),
                conv_idx=args.conv_idx, damp=args.damp, round_to=args.round_to, floor=args.floor,
                resplit=not args.no_resplit, win_per_corpus=args.win_per_corpus, chunk_per_corpus=args.chunk_per_corpus,
                batch=args.batch, seed=args.seed, per_corpus=per, ridge_used=ridge,
                rank_sum_L0_11=early, rank_sum_L12_23=late,
                conv_ranks={f"conv{i}": int(ranks[("conv", i)]) for i in args.conv_idx},
                resplit_scale_median=(float(np.median(list(scales.values()))) if scales else None),
                timing_sec={k: round(v, 2) for k, v in T.items()}, out_bin=args.out_bin,
                teacher_wavlm=args.teacher_wavlm, teacher_diar=args.teacher_diar)
    json.dump(meta, open(os.path.join(dump_dir, "meta.json"), "w"), indent=2)
    print(f"[build {args.tag}] {P_lr/1e6:.2f}M (real {P_full/P_lr:.2f}x) | ranks L0-11 {early} L12-23 {late} | "
          f"conv {[int(ranks[('conv', i)]) for i in args.conv_idx]} | " +
          " ".join(f"{k}={v:.1f}s" for k, v in T.items()), flush=True)

    # assemble a DiariZen experiment directory (compressed WavLM + teacher head) for eval_tf.sh
    del dense
    torch.cuda.empty_cache()
    m = build_diar_model(args.out_bin, args.teacher_diar, HEAD_LARGE)
    torch.save(m.state_dict(), write_exp_scaffold(args.exp_dir, args.out_bin, HEAD_LARGE))
    print(f"[assemble] {args.exp_dir}", flush=True)


if __name__ == "__main__":
    main()
