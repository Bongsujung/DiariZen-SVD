#!/usr/bin/env python3
"""Split the DiariZen compound lists into one directory per corpus (the layout of the paper's Table 1).

    data/<CORPUS>/train/    {wav.scp, rttm, all.uem}   fine-tuning / calibration recordings
    data/<CORPUS>/dev/      {wav.scp, rttm, all.uem}   validation (NOTSOFAR-1 has none)
    data/<CORPUS>/eval/     {wav.scp, rttm}            evaluation
    data/<CORPUS>/recover/  {wav.scp, rttm, all.uem}   the 50 % of train used for recovery

    data/_pooled/{train,dev,recover}/                  derived: the corpora concatenated in CORPORA order,
                                                       what the calibration and recovery loaders consume

Source is a DiariZen ``recipes/diar_ssl/data`` directory with ``compound/{train,dev,test/<corpus>}`` and
``compound_50/train``.  The corpus of a recording is inferred by ``diarizen_svd.data.corpus.corpus_of``.
The recording counts are checked against Table 1 of the paper and the script fails if they differ.

Usage:  python scripts/prepare_data.py --src /path/to/DiariZen/recipes/diar_ssl/data [--out data]
"""
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarizen_svd.data.corpus import CORPORA, corpus_of

# Table 1 of the paper: recordings per corpus (train, dev, eval, recover)
TABLE1 = {"AMI": (134, 18, 16, 65), "AISHELL4": (173, 18, 20, 92), "AliMeeting": (209, 8, 20, 92),
          "RAMC": (289, 19, 43, 137), "VoxConverse": (174, 42, 232, 98), "MSDWild": (2476, 177, 490, 1255),
          "NOTSOFAR": (526, 0, 160, 251)}
SPLITS = ["train", "dev", "eval", "recover"]


def read_lines(path):
    return [l for l in open(path)] if os.path.exists(path) else []


def split_by_corpus(scp_lines):
    by = defaultdict(list)
    for l in scp_lines:
        if l.strip():
            p = l.split()
            by[corpus_of(p[0], p[-1])].append(l)
    return by


def write_split(out_dir, scp, rttm, uem):
    os.makedirs(out_dir, exist_ok=True)
    ids = {l.split()[0] for l in scp}
    open(os.path.join(out_dir, "wav.scp"), "w").writelines(scp)
    open(os.path.join(out_dir, "rttm"), "w").writelines(l for l in rttm if len(l.split()) > 1 and l.split()[1] in ids)
    if uem is not None:
        open(os.path.join(out_dir, "all.uem"), "w").writelines(l for l in uem if l.split() and l.split()[0] in ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="DiariZen recipes/diar_ssl/data (has compound/ and compound_50/)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    args = ap.parse_args()
    S = args.src

    # pooled sources -> per corpus
    src = {"train": f"{S}/compound/train", "dev": f"{S}/compound/dev", "recover": f"{S}/compound_50/train"}
    per = {c: {} for c in CORPORA}
    for split, d in src.items():
        by = split_by_corpus(read_lines(f"{d}/wav.scp"))
        rttm, uem = read_lines(f"{d}/rttm"), read_lines(f"{d}/all.uem")
        unk = sum(len(v) for k, v in by.items() if k not in CORPORA)
        if unk:
            sys.exit(f"[prepare_data] {unk} recordings of {d}/wav.scp could not be assigned to a corpus "
                     f"(adjust diarizen_svd/data/corpus.py:corpus_of)")
        for c in CORPORA:
            per[c][split] = (by.get(c, []), rttm, uem)
    for c in CORPORA:                                   # evaluation lists are already per corpus in the source
        d = f"{S}/compound/test/{c}"
        per[c]["eval"] = (read_lines(f"{d}/wav.scp"), read_lines(f"{d}/rttm"), None)

    # write per-corpus directories and the pooled views
    counts, ok = {}, True
    pooled = {s: ([], [], []) for s in ("train", "dev", "recover")}
    for c in CORPORA:
        counts[c] = []
        for s in SPLITS:
            scp, rttm, uem = per[c][s]
            write_split(f"{args.out}/{c}/{s}", scp, rttm, uem)
            counts[c].append(len(scp))
            if s in pooled:
                ids = {l.split()[0] for l in scp}
                pooled[s][0].extend(scp)
                pooled[s][1].extend(l for l in rttm if len(l.split()) > 1 and l.split()[1] in ids)
                pooled[s][2].extend(l for l in uem if l.split() and l.split()[0] in ids)
    for s, (scp, rttm, uem) in pooled.items():
        write_split(f"{args.out}/_pooled/{s}", scp, rttm, uem)

    # report against Table 1
    print(f"{'corpus':<12}" + "".join(f"{s:>10}" for s in SPLITS))
    for c in CORPORA:
        row = "".join(f"{n:>7}{'   ' if n == e else ' !!'}" for n, e in zip(counts[c], TABLE1[c]))
        ok &= tuple(counts[c]) == TABLE1[c]
        print(f"{c:<12}{row}")
    tot = [sum(counts[c][i] for c in CORPORA) for i in range(4)]
    print(f"{'total':<12}" + "".join(f"{n:>7}   " for n in tot) + "   (paper: 3981 / 282 / 981 / 1990)")
    if not ok:
        sys.exit("[prepare_data] recording counts differ from Table 1 of the paper (rows marked !!)")
    print(f"[prepare_data] {args.out}/<corpus>/{{train,dev,eval,recover}} and {args.out}/_pooled/ written; counts match Table 1")


if __name__ == "__main__":
    main()
