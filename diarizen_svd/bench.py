#!/usr/bin/env python3
"""Efficiency measurement primitives: model construction from a spec, MACs profiling, throughput timing.

A model spec is  ``<path.bin>``  (a {config, state_dict} checkpoint, dense or low-rank) or
``config:<name>``  (a predefined config of ``diarizen/models/module/wavlm_config.py``, e.g. the
released pruned encoder ``wavlm_large_s80_md`` -- note that this gives the *architecture* only, which
is all that params / MACs / speed depend on).

MACs are counted with the DeepSpeed FlopsProfiler [github.com/microsoft/DeepSpeed] on one second of
16 kHz audio, which is the convention of the DiariZen pruning recipe and of Han et al. [TASLP 2026,
Table II] (they report 17.8G for dense WavLM-Large and 3.8G at 80 % sparsity).
"""
import time

import torch

from diarizen.models.module.wavlm_config import get_config
from diarizen.models.module.wav2vec2.model import wav2vec2_model

from diarizen_svd.experiment import load_wavlm_from_bin


def build(spec):
    """``name=<path.bin>`` / ``name=config:<config_name>`` right-hand side -> a WavLM encoder."""
    if spec.startswith("config:"):
        return wav2vec2_model(**get_config(spec[len("config:"):]))
    return load_wavlm_from_bin(spec)[0]


def profile(w, secs=1.0):
    """Parameters and MACs of one encoder, split into CNN front-end and Transformer."""
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    w = w.eval().cpu().float()
    pm = lambda mod: sum(v.numel() for v in mod.parameters()) / 1e6
    pr = FlopsProfiler(w)
    pr.start_profile()
    with torch.no_grad():
        w.extract_features(torch.randn(1, int(16000 * secs)))
    tot = pr.get_total_macs() / 1e9
    cnn = sum(getattr(mm, "__macs__", 0) for mm in w.feature_extractor.modules()) / 1e9
    pr.end_profile()
    return dict(params_M=round(pm(w), 2), params_cnn_M=round(pm(w.feature_extractor), 2),
                params_transformer_M=round(pm(w.encoder), 2), macs_G=round(tot, 3),
                macs_cnn_G=round(cnn, 3), macs_transformer_G=round(tot - cnn, 3))


def bench(model, bs, n_samples, secs, iters, dev):
    """Throughput in audio-seconds per wall-clock second at batch size ``bs`` (3 warm-up iterations)."""
    x = torch.randn(bs, n_samples, device=dev)
    with torch.no_grad():
        for _ in range(3):
            model.extract_features(x)                      # warm-up
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model.extract_features(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return iters * bs * secs / dt                          # audio-seconds per wall-second
