#!/usr/bin/env python3
"""Which matrices of a WavLM-Large encoder are compressed, and how they are named.

The compressed set is fixed here and is the "150 matrices in total" of the paper (§2):
the six linear projections of every Transformer layer (24 x 6 = 144) plus CNN layers 1-6.
``conv0`` (1 -> 512 channels, K = 10, im2col matrix 512 x 10), the positional convolution, the feature
projection, all norms and all biases are never factorized.
"""

# The six linear maps of every Transformer layer (WavLM-Large: 24 layers x 6 = 144 matrices) ...
TARGETS = ["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj",
           "feed_forward.intermediate_dense", "feed_forward.output_dense"]
# ... and six of the seven CNN front-end layers.  conv0 (1 -> 512 channels, K=10, 5.1K params)
# is kept dense: its im2col matrix is 512 x 10 and has nothing to gain from a factorization.
CONV_IDX = [1, 2, 3, 4, 5, 6]


def target_modules(wavlm, conv_idx=CONV_IDX):
    """{key: nn.Module} for every compressed matrix.  key = ('lin', layer, target) | ('conv', idx)."""
    mods = {}
    layers = wavlm.encoder.transformer.layers
    for li in range(len(layers)):
        for t in TARGETS:
            mods[("lin", li, t)] = layers[li].get_submodule(t)
    for i in conv_idx:
        mods[("conv", i)] = wavlm.feature_extractor.conv_layers[i].conv
    return mods


def key_name(key) -> str:
    return f"lin.L{key[1]}.{key[2].split('.')[-1]}" if key[0] == "lin" else f"conv.L{key[1]}.conv"


def sort_keys(keys):
    return sorted(keys, key=lambda k: (0 if k[0] == "conv" else 1, k[1], k[2] if k[0] == "lin" else ""))
