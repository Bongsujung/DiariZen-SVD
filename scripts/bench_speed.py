#!/usr/bin/env python3
"""Inference speed-up of WavLM encoders under the protocol of Han et al. [TASLP 2026, Table II]:

  "Inference speedups on both GPU and CPU are reported relative to the unpruned model ... with the
   input batch size tuned to maximize utilization of each device computational capacity."

Only the encoder is timed (their Params/MACs scope) on a 16 s window (their inference segment
length).  The batch size is swept per model and device and the best throughput (audio seconds per
wall-clock second) is kept; the speed-up is the throughput ratio to the first (dense) model.
Timing lives in :func:`diarizen_svd.bench.bench`; this is the command-line front end.

Usage:
  python scripts/bench_speed.py --device cuda --batches 1,2,4,8,16,32,64 --iters 12 --models ...
  OMP_NUM_THREADS=8 python scripts/bench_speed.py --device cpu --batches 1,2,4,8 --iters 4 --models ...
Model specs as in bench_macs.py (name=<path.bin> | name=config:<config_name>); the first is the reference.
"""
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarizen_svd.bench import build, bench


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="first = reference (dense) model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--secs", type=float, default=16.0)
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = torch.device(args.device)
    N = int(args.secs * 16000)
    rows = []
    for spec in args.models:
        name, src = spec.split("=", 1)
        m = build(src).eval().to(dev).float()
        p = sum(v.numel() for v in m.parameters()) / 1e6
        best, best_bs = 0.0, None
        for bs in [int(b) for b in args.batches.split(",")]:
            try:
                th = bench(m, bs, N, args.secs, args.iters, dev)
            except RuntimeError as e:                       # out of memory: stop the sweep
                print(f"  {name} bs={bs}: {str(e)[:60]}", flush=True)
                break
            print(f"  {name} bs={bs}: {th:.1f} audio-s/s", flush=True)
            if th > best:
                best, best_bs = th, bs
        rows.append(dict(model=name, source=src, params_M=round(p, 2), best_batch=best_bs, throughput=round(best, 2)))
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    base = rows[0]["throughput"]
    for r in rows:
        r["speedup"] = round(r["throughput"] / base, 2)
    print(f"\n=== {args.device.upper()} | {args.secs:.0f} s window | speed-up vs {rows[0]['model']} ===")
    for r in rows:
        print(f"{r['model']:<20} {r['params_M']:>8.2f}M  bs={r['best_batch']:<3} {r['throughput']:>9.1f} audio-s/s  {r['speedup']:>5.2f}x")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
