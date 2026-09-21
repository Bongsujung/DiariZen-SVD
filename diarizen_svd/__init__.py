"""DiariZen-SVD: post-hoc low-rank (SVD) compression of a fine-tuned WavLM diarization encoder.

Depends on an unmodified DiariZen (https://github.com/BUTSpeechFIT/DiariZen) installation; the only
model-side addition is ``diarizen_svd.nn.model.Model``, whose ``load_wavlm`` wraps the target layers of
a dense WavLM with the factor pairs of ``diarizen_svd.nn.lowrank`` when the checkpoint config asks for
it.

Only the light-weight layer definitions are re-exported here; everything that needs DiariZen itself
(``nn.model``, ``experiment``) must be imported explicitly so that ``import diarizen_svd`` stays cheap.
"""
from .nn.lowrank import LowRankLinear, LowRankConv1d, apply_lowrank, build_from_config  # noqa: F401
