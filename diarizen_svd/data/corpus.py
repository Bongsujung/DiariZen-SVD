#!/usr/bin/env python3
"""Corpus bookkeeping for the seven-corpus compound set.

Calibration and recovery are balanced *per corpus* rather than drawn from the pooled training list,
because the compound set is dominated by MSDWild (2476 of 3981 training recordings).  Every helper
here works on the Kaldi-style ``wav.scp`` lists of DiariZen ``recipes/diar_ssl/data``.
"""
import os
import random
import re
from collections import defaultdict

CORPORA = ["AMI", "AISHELL4", "AliMeeting", "RAMC", "VoxConverse", "MSDWild", "NOTSOFAR"]


def corpus_of(rec_id: str, path: str) -> str:
    """Map a recording of the compound set to its corpus (from the wav path / recording-id pattern)."""
    if "/msdwild/" in path: return "MSDWild"
    if "/magicdata_ramc/" in path: return "RAMC"
    if "/voxconverse/" in path: return "VoxConverse"
    if "/notsofar/" in path: return "NOTSOFAR"
    if "AMI_AliMeeting_AISHELL4" in path:
        if rec_id.startswith("R"): return "AliMeeting"
        if rec_id.startswith("2020"): return "AISHELL4"
        if re.match(r"^(ES|TS|IS|EN|IN|IB)", rec_id): return "AMI"
    return "UNK"


def read_scp(scp: str):
    """Kaldi-style wav.scp -> [(rec_id, path)]."""
    out = []
    for line in open(scp):
        if line.strip():
            p = line.split()
            out.append((p[0], p[-1]))
    return out


def per_corpus_lists(scp: str, seed: int = 0):
    """{corpus: [(rec_id, path), ...]} in CORPORA order, each list shuffled with a fresh Random(seed).

    Re-seeding per corpus makes the calibration file order identical to the runs of the paper."""
    by = defaultdict(list)
    for rec_id, path in read_scp(scp):
        by[corpus_of(rec_id, path)].append((rec_id, path))
    out = {}
    for c in CORPORA:
        lst = list(by.get(c, []))
        random.Random(seed).shuffle(lst)
        out[c] = lst
    return out


def balanced_dev_subset(dev_dir, K, exp_dir, seed=0):
    """K random recordings per corpus from <dev_dir>/{wav.scp,rttm,all.uem} -> (scp, rttm, uem) under <exp_dir>/_devsub.*"""
    os.makedirs(exp_dir, exist_ok=True)
    scp = [l for l in open(f"{dev_dir}/wav.scp") if l.strip()]
    rttm = open(f"{dev_dir}/rttm").readlines()
    uem = {l.split()[0]: l for l in open(f"{dev_dir}/all.uem") if l.strip()}
    by = defaultdict(list)
    for l in scp:
        by[corpus_of(l.split()[0], l.split()[-1])].append(l)
    rng, keep, recs = random.Random(seed), [], set()
    for c, lines in by.items():
        if c == "UNK":
            continue
        rng.shuffle(lines)
        keep += lines[:K]
        recs |= {s.split()[0] for s in lines[:K]}
    pre = os.path.join(exp_dir, "_devsub")
    open(pre + ".scp", "w").writelines(keep)
    open(pre + ".rttm", "w").writelines([l for l in rttm if len(l.split()) > 1 and l.split()[1] in recs])
    open(pre + ".uem", "w").writelines([uem[r] for r in recs if r in uem])
    print(f"[dev subset] {len(recs)} recordings ({K} per corpus x {len([c for c in by if c != 'UNK'])} corpora)", flush=True)
    return pre + ".scp", pre + ".rttm", pre + ".uem"
