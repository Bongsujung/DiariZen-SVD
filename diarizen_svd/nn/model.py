# Licensed under the MIT license.
"""DiariZen EEND model whose WavLM loader understands low-rank checkpoints.

``diarizen.models.eend.model_wavlm_conformer.Model`` (WavLM -> learned layer weighting -> Conformer ->
powerset head; Han et al., ICASSP 2025) is subclassed and only ``load_wavlm`` is overridden: a
checkpoint whose config carries ``lowrank_ranks`` / ``cnn_lowrank_ranks`` is built as the dense
architecture and its target layers are replaced by factor pairs (``diarizen_svd.nn.lowrank``) before
the state_dict is loaded.  Dense checkpoints and predefined configs behave exactly as upstream.

Point ``[model] path`` of a DiariZen experiment config at ``diarizen_svd.nn.model.Model`` (see
``conf/wavlm_large_lowrank.toml``); ``infer_avg.py`` and the training scripts of DiariZen then use
this class through ``diarizen.utils.instantiate``.
"""
import os
import torch

from diarizen.models.eend.model_wavlm_conformer import Model as _DiariZenModel
from diarizen.models.module.wav2vec2.model import wav2vec2_model
from diarizen.models.module.wavlm_config import get_config
from diarizen_svd.nn.lowrank import build_from_config


class Model(_DiariZenModel):
    """DiariZen ``Model`` with low-rank-aware WavLM loading."""

    def load_wavlm(self, source: str, *args, **kwargs):
        if os.path.isfile(source):
            ckpt = torch.load(source, map_location="cpu")
            if "config" not in ckpt or "state_dict" not in ckpt:
                raise ValueError("Checkpoint must contain 'config' and 'state_dict'.")
            for k, v in ckpt["config"].items():
                if "prune" in k and v is not False:
                    raise ValueError(f"Pruning must be disabled. Found: {k}={v}")
            model = build_from_config(ckpt["config"], wav2vec2_model)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            return model
        return wav2vec2_model(**get_config(source))
