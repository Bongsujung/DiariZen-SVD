"""Runtime model code: the low-rank layers and the DiariZen model subclass that loads them.

Intentionally empty of re-exports.  ``lowrank`` needs only ``torch.nn``, while ``model`` pulls in
DiariZen and pyannote.audio; re-exporting ``model`` here would make ``import diarizen_svd`` drag the
whole diarization stack in.  Import what you need explicitly:

    from diarizen_svd.nn.lowrank import LowRankLinear      # light
    from diarizen_svd.nn.model import Model                # heavy (DiariZen)
"""
