One directory per corpus, one directory per split (the layout of Table 1 of the paper):

```
data/<CORPUS>/train/    wav.scp  rttm  all.uem     fine-tuning / calibration recordings
data/<CORPUS>/dev/      wav.scp  rttm  all.uem     validation (NOTSOFAR-1 has none)
data/<CORPUS>/eval/     wav.scp  rttm              evaluation (eval_tf.sh)
data/<CORPUS>/recover/  wav.scp  rttm  all.uem     the 50 % of train used for recovery
data/_pooled/{train,dev,recover}/                  derived: all corpora concatenated, consumed by the calibration and recovery loaders
```

CORPUS is one of AMI, AISHELL4, AliMeeting, RAMC, VoxConverse, MSDWild, NOTSOFAR. Files follow the DiariZen
convention (`recipes/diar_ssl/data` of https://github.com/BUTSpeechFIT/DiariZen): `wav.scp` lines are
`<recording-id> <path>`, `rttm` is standard RTTM, `all.uem` is `<recording-id> <channel> <start> <end>`.

`scripts/prepare_data.py --src <DiariZen>/recipes/diar_ssl/data` builds this tree from DiariZen's pooled
`compound/` and `compound_50/` lists and checks the recording counts (3981 / 282 / 981 / 1990) against the
paper. The corpus of a recording is inferred from its path / id by `diarizen_svd/data/corpus.py:corpus_of`;
adjust that function if your audio lives elsewhere.
