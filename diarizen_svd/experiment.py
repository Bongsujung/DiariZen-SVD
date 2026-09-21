#!/usr/bin/env python3
"""The DiariZen integration layer: loading encoders, assembling the EEND model, the diarization loss,
and writing the minimal experiment directory that DiariZen's ``infer_avg.py`` expects.

This recipe is one branch of the DiariZen code base
[Han et al., "Leveraging Self-Supervised Learning for Speaker Diarization", ICASSP 2025;
 https://github.com/BUTSpeechFIT/DiariZen].  It replaces the structured-pruning compression of
``recipes/diar_ssl_pruning`` [Han et al., "Efficient and Robust Speaker Diarization via Structured
Pruning of Self-Supervised Models", IEEE/ACM TASLP 34:1903-1914, 2026] with a post-hoc low-rank
(SVD) factorization of the task-fine-tuned WavLM-Large encoder.  Everything downstream of the
encoder -- the EEND head, the powerset loss [Plaquet & Bredin, Interspeech 2023], VBx clustering
[Landini et al., CSL 2022] and DER scoring (dscore) -- is the unmodified DiariZen / pyannote.audio
pipeline; ``dataset.py`` and ``infer_avg.py`` are verbatim copies of DiariZen ``recipes/diar_ssl``.

DiariZen itself is used unmodified.  The low-rank layers live in ``diarizen_svd/nn/lowrank.py`` and
``diarizen_svd/nn/model.py`` subclasses the DiariZen EEND model so that a checkpoint whose config
carries ``lowrank_ranks`` / ``cnn_lowrank_ranks`` is built as the dense architecture and wrapped post
hoc.
"""
import os

import torch

from diarizen_svd.paths import ROOT

# EEND head of the WavLM-Large DiariZen model (recipes/diar_ssl/conf/wavlm_updated_conformer.toml)
HEAD_LARGE = dict(wavlm_layer_num=25, wavlm_feat_dim=1024, attention_in=256, ffn_hidden=1024,
                  num_head=4, num_layer=4, dropout=0.1, chunk_size=8, use_posi=False,
                  output_activate_function=False, selected_channel=0, max_speakers_per_chunk=4)


# ----------------------------------------------------------------------------- models and loss
def load_wavlm_from_bin(path: str):
    """Instantiate a (dense or low-rank) WavLM from a ``{config, state_dict}`` checkpoint."""
    from diarizen.models.module.wav2vec2.model import wav2vec2_model
    from diarizen_svd.nn.lowrank import build_from_config
    ck = torch.load(path, map_location="cpu")
    m = build_from_config(ck["config"], wav2vec2_model)
    m.load_state_dict(ck["state_dict"], strict=False)
    return m, ck


def build_diar_model(wavlm_src: str, ft_diar: str, head: dict = HEAD_LARGE):
    """DiariZen ``Model`` = WavLM from ``wavlm_src`` + trained head (all non-WavLM weights) from ``ft_diar``."""
    from diarizen_svd.nn.model import Model
    m = Model(wavlm_src=wavlm_src, **head)
    ft = torch.load(ft_diar, map_location="cpu")
    ft = ft.get("state_dict", ft) if isinstance(ft, dict) else ft
    res = m.load_state_dict({k: v for k, v in ft.items() if not k.startswith("wavlm_model.")}, strict=False)
    assert not res.unexpected_keys, f"unexpected head keys: {res.unexpected_keys[:5]}"
    return m


def diar_loss(model, xs, target):
    """Permutation-invariant powerset NLL of DiariZen (pyannote.audio permutate + nll_loss)."""
    from pyannote.audio.utils.permutation import permutate
    from pyannote.audio.utils.loss import nll_loss
    yp = model(xs)
    ml = model.powerset.to_multilabel(yp)
    pt, _ = permutate(ml, target)
    ptp = model.powerset.to_powerset(pt.float())
    return nll_loss(yp, torch.argmax(ptp, dim=-1))


# ----------------------------------------------------------------------------- experiment directories
def write_exp_scaffold(exp_dir: str, wavlm_bin: str, head: dict = HEAD_LARGE,
                       template: str = f"{ROOT}/conf/wavlm_large_lowrank.toml"):
    """Create the minimal DiariZen experiment directory that ``infer_avg.py`` expects:
    ``<name>.toml`` (model config), ``val_metric_summary.lst`` (one dummy epoch) and
    ``checkpoints/epoch_0001/`` (filled by the caller)."""
    os.makedirs(os.path.join(exp_dir, "checkpoints", "epoch_0001"), exist_ok=True)
    name = os.path.basename(exp_dir.rstrip("/"))
    # repository-relative when the encoder lives inside the repo, so the directory stays portable
    # (eval_tf.sh and run_stage.sh cd to the repository root before calling infer_avg.py)
    src = os.path.abspath(wavlm_bin)
    src = os.path.relpath(src, ROOT) if src.startswith(ROOT + os.sep) else src
    conf = open(template).read().replace("{WAVLM_SRC}", src)
    for k, v in head.items():
        conf = conf.replace("{" + k.upper() + "}", str(v).lower() if isinstance(v, bool) else str(v))
    open(os.path.join(exp_dir, f"{name}.toml"), "w").write(conf)
    open(os.path.join(exp_dir, "val_metric_summary.lst"), "w").write(
        " Validation Loss/DER on epoch 1: 0.000 / 0.000\n")
    return os.path.join(exp_dir, "checkpoints", "epoch_0001", "pytorch_model.bin")
