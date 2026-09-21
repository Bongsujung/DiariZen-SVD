#!/usr/bin/env python3
"""DiariZen ``DiarizationDataset`` loaders and the per-corpus file subsets they need.

``DiarizationDataset`` takes one (scp, rttm, uem) triple per loader, so a corpus-balanced sweep first
writes a small per-corpus subset of the pooled lists (:func:`write_corpus_subset`) and then builds one
loader per corpus.
"""
from functools import partial


def write_corpus_subset(prefix: str, corpus: str, recs, rttm_lines, uem_lines):
    """Write per-corpus wav.scp / rttm / uem files (DiarizationDataset needs one file set per loader)."""
    ids = {r for r, _ in recs}
    scp = f"{prefix}.{corpus}.scp"
    rttm = f"{prefix}.{corpus}.rttm"
    uem = f"{prefix}.{corpus}.uem"
    with open(scp, "w") as f:
        f.write("".join(f"{r}\t{p}\n" for r, p in recs))
    with open(rttm, "w") as f:
        f.writelines(l for l in rttm_lines if len(l.split()) > 1 and l.split()[1] in ids)
    with open(uem, "w") as f:
        f.writelines(l for l in uem_lines if l.split() and l.split()[0] in ids)
    return scp, rttm, uem


def make_loader(model, scp, rttm, uem, batch, chunk_size=8, chunk_shift=8, shuffle=True, num_workers=4):
    """DiariZen ``DiarizationDataset`` loader with the frame geometry of ``model`` (8 s chunks by default)."""
    from dataset import DiarizationDataset, _collate_fn           # DiariZen recipes/diar_ssl/dataset.py (copied)
    from torch.utils.data import DataLoader
    _, rfd, rfs = model.get_rf_info
    nf = int((chunk_size - rfd) / rfs) + 1
    ds = DiarizationDataset(scp_file=scp, rttm_file=rttm, uem_file=uem, chunk_size=chunk_size,
                            chunk_shift=chunk_shift, sample_rate=16000, model_num_frames=nf,
                            model_rf_duration=rfd, model_rf_step=rfs)
    return DataLoader(ds, collate_fn=partial(_collate_fn, max_speakers_per_chunk=4), batch_size=batch,
                      shuffle=shuffle, num_workers=num_workers, drop_last=shuffle)
