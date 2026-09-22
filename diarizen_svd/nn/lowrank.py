# Licensed under the MIT license.
"""Low-rank (SVD-factorized) layers for a WavLM encoder, applied post hoc to a dense model.

This module is self-contained: it does not touch the WavLM constructor, so a compressed encoder is
a dense ``wav2vec2_model(**config)`` whose target layers are replaced after construction.  The
checkpoint config carries only two extra keys, consumed by :func:`build_from_config` /
``diarizen_svd.nn.model.Model.load_wavlm``:

    lowrank_ranks     : list (one dict per Transformer layer) or dict {leaf path -> rank}, e.g.
                        [{"attention.q_proj": 96, ..., "feed_forward.output_dense": 64}, ...]
    cnn_lowrank_ranks : {conv layer index -> rank} for the CNN front-end

The factors themselves are computed offline (diarizen_svd/compress/factorize.py; whitened SVD after) and loaded through the normal
state_dict; this file only defines the compressed forward passes.

    LowRankLinear : nn.Linear W (d_out x d_in)  ->  y = A (B x) + b,   A: d_out x r,  B: r x d_in
    LowRankConv1d : nn.Conv1d W (C_out x C_in x K) -> conv_A(1x1) ( conv_B(K) x ),
                    i.e. the im2col matrix (C_out x C_in K) factorized as A (C_out x r) B (r x C_in K)
"""
from typing import Dict, List, Optional, Union

import torch.nn as nn

CONFIG_KEYS = ("lowrank_ranks", "cnn_lowrank_ranks")


class LowRankLinear(nn.Module):
    """Drop-in replacement for nn.Linear holding a low-rank factorization W ~= A @ B."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lr_B = nn.Linear(in_features, rank, bias=False)    # B: (r, d_in)
        self.lr_A = nn.Linear(rank, out_features, bias=bias)    # A: (d_out, r) (+ bias)

    @classmethod
    def from_linear(cls, base: nn.Linear, rank: int):
        return cls(base.in_features, base.out_features, rank, bias=base.bias is not None)

    def forward(self, x):
        return self.lr_A(self.lr_B(x))


class LowRankConv1d(nn.Module):
    """Conv1d W (C_out, C_in, K) ~= A (C_out, r, 1) * B (r, C_in, K);  forward = conv_A(conv_B(x))."""

    def __init__(self, in_ch: int, out_ch: int, k: int, rank: int, stride: int, padding: int, bias: bool):
        super().__init__()
        self.conv_B = nn.Conv1d(in_ch, rank, k, stride=stride, padding=padding, bias=False)
        self.conv_A = nn.Conv1d(rank, out_ch, 1, bias=bias)

    @classmethod
    def from_conv(cls, base: nn.Conv1d, rank: int):
        return cls(base.in_channels, base.out_channels, base.kernel_size[0], int(rank),
                   base.stride[0], base.padding[0], base.bias is not None)

    def forward(self, x):
        return self.conv_A(self.conv_B(x))


def wrap_layer_lowrank(layer: nn.Module, ranks: Dict[str, int]) -> None:
    """Replace the nn.Linear submodules of one Transformer layer named in ``ranks`` by LowRankLinear.

    ``ranks`` maps a leaf path relative to the layer (e.g. "attention.q_proj") to a rank; a rank of
    0 / None keeps the dense layer, and paths that do not exist are skipped."""
    for path, r in ranks.items():
        if not r:
            continue
        *mods, leaf = path.split(".")
        obj = layer
        for m in mods:
            obj = getattr(obj, m, None)
            if obj is None:
                break
        if obj is None:
            continue
        base = getattr(obj, leaf, None)
        if isinstance(base, nn.Linear) and not isinstance(base, LowRankLinear):
            setattr(obj, leaf, LowRankLinear.from_linear(base, int(r)))


def wrap_cnn_lowrank(wavlm: nn.Module, ranks: Union[Dict[int, int], List[int]]) -> None:
    """Replace ``feature_extractor.conv_layers[idx].conv`` by LowRankConv1d for every idx in ``ranks``
    (rank <= 0 keeps the dense convolution).  Weights are loaded afterwards from the state_dict."""
    cl = wavlm.feature_extractor.conv_layers
    for idx, r in (ranks.items() if isinstance(ranks, dict) else enumerate(ranks)):
        idx = int(idx)
        if not r or int(r) <= 0:
            continue
        cl[idx].conv = LowRankConv1d.from_conv(cl[idx].conv, int(r))


def apply_lowrank(wavlm: nn.Module, lowrank_ranks=None, cnn_lowrank_ranks=None) -> nn.Module:
    """Turn a dense WavLM into its low-rank counterpart (empty factors, to be loaded from a state_dict).

    lowrank_ranks     : list of per-layer dicts (index = Transformer layer) or one dict for all layers
    cnn_lowrank_ranks : {conv index -> rank}
    """
    if lowrank_ranks:
        layers = wavlm.encoder.transformer.layers
        for li in range(len(layers)):
            per = lowrank_ranks[li] if isinstance(lowrank_ranks, (list, tuple)) else lowrank_ranks
            wrap_layer_lowrank(layers[li], per)
    if cnn_lowrank_ranks:
        wrap_cnn_lowrank(wavlm, cnn_lowrank_ranks)
    return wavlm


def split_config(config: dict):
    """(dense wav2vec2 config, lowrank_ranks, cnn_lowrank_ranks) from a checkpoint config."""
    cfg = dict(config)
    return cfg, cfg.pop("lowrank_ranks", None), cfg.pop("cnn_lowrank_ranks", None)


def build_from_config(config: dict, model_fn) -> nn.Module:
    """``model_fn(**dense_config)`` followed by :func:`apply_lowrank` with the ranks found in ``config``."""
    cfg, lin, cnn = split_config(config)
    return apply_lowrank(model_fn(**cfg), lin, cnn)


def lowrank_modules(wavlm: nn.Module):
    """{name: module} of every LowRankLinear / LowRankConv1d in the encoder."""
    return {n: m for n, m in wavlm.named_modules() if isinstance(m, (LowRankLinear, LowRankConv1d))}
