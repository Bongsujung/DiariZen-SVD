# DiariZen-SVD

Post-training low-rank compression of the fine-tuned WavLM-Large encoder of a
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen) speaker-diarization system, as an alternative to structured
pruning. Bi-directional whitened SVD with the CNN front-end included via im2col, one parameter budget for all
150 matrices, and sequential LoRA recovery. DiariZen itself is used unmodified.

Paper: *Bi-directional Whitened Low-Rank Compression of Task-Fine-Tuned WavLM for Speaker Diarization*
(ICASSP 2027, submitted).

## Installation

From the repository root:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt                         # torch 2.1.1+cu121 pinned here; install it before DiariZen

# DiariZen, unmodified (tested at commit 844f555)
git clone https://github.com/BUTSpeechFIT/DiariZen.git ../DiariZen && pushd ../DiariZen
pip install -r requirements.txt && pip install -e .
(cd pyannote-audio && pip install -e ".[dev,testing]")
git submodule update --init                             # dscore
popd && ln -s ../DiariZen/dscore dscore
```

Check the result before going on — a different torch or numpy means DiariZen pulled its own in:

```bash
python -c "import torch, numpy, diarizen, pyannote.audio as pa; \
  print(torch.__version__, numpy.__version__, pa.__version__, torch.cuda.is_available()); print(diarizen.__file__)"
# 2.1.1+cu121 1.26.4 3.1.1 True
# .../DiariZen/diarizen/__init__.py
```

If `/tmp` or `~/.cache/pip` are on a small partition, point pip elsewhere first:
`export TMPDIR=/path/with/space PIP_CACHE_DIR=/path/with/space/pip-cache`.

One CUDA GPU is needed for everything except scoring; the paper used a single RTX 4090 (24 GB). Building at
one ratio writes an 8.3 GB calibration cache under `exp/stage1/cache/` that every further ratio reuses.

## How to run

### Assets

Paths are the defaults of `conf/run.toml`; every one of them can be pointed elsewhere there. Scripts fail
early, naming the flag to use, if something is missing.

| Asset | Needed by | Where from |
|:--|:--|:--|
| `pretrained/eend_wavlm_large_compound7_avg5.bin` | stages 0–2 | the released archive, or DiariZen `recipes/diar_ssl/run_stage.sh` on the compound set |
| `pretrained/wavlm-large-converted.bin` | stage 0 only — its **architecture config**, never its weights | DiariZen `recipes/diar_ssl_pruning/convert_wavlm_from_hf.py` on `microsoft/wavlm-large` |
| `pretrained/pyannote/wespeaker-voxceleb-resnet34-LM/` | evaluation | Hugging Face `pyannote/wespeaker-voxceleb-resnet34-LM` |
| `pretrained/diarizen-wavlm-base-s80-md/` | evaluation — only its `plda/` | Hugging Face, any DiariZen model, e.g. `BUT-FIT/diarizen-wavlm-base-s80-md` |
| `dscore/` | evaluation | the DiariZen submodule linked above |
| `data/<corpus>/{train,dev,eval,recover}/` | all | `scripts/prepare_data.py`, below |

Keep `pyannote/` in the embedding path: pyannote.audio picks its loader from the path string, and a path
without it is handed to the ONNX loader, which cannot read this checkpoint.

### Data

You need the seven corpora themselves and DiariZen's list files for them
(`recipes/diar_ssl/data/compound`, `compound_50`). This splits those pooled lists into one directory per
corpus and per split, and checks the recording counts against Table 1 of the paper:

```bash
python scripts/prepare_data.py --src ../DiariZen/recipes/diar_ssl/data
# corpus           train       dev      eval   recover
# AMI               134        18        16        65
# ...
# total            3981       282       981      1990      (paper: 3981 / 282 / 981 / 1990)
```

It exits non-zero if the counts differ. The corpus of a recording is inferred from its path by
`diarizen_svd/data/corpus.py:corpus_of`; adjust that function if your audio lives elsewhere.

### Stages

Every setting lives in `conf/run.toml`; `--method` and `--ratio` override `[compress]` for one call.
All outputs go to `exp/stage<N>/`, and each stage reads only the outputs of the previous ones.

```bash
bash run_stage.sh 0                              # teacher extraction                seconds
bash run_stage.sh 1 --method obdllm --ratio 5    # compression, then evaluation       ~7 min + hours
bash run_stage.sh 2 --method obdllm --ratio 5    # recovery, averaging, evaluation    17.5 h on one RTX 4090
bash run_stage.sh 3 --method obdllm --ratio 5    # Params / MACs / speed-up           minutes
```

`--method` is `obdllm` (bi-directional whitened SVD, the paper's system), `svdllm` or `fwsvd` (the Table 3
baselines); `--ratio` is ρ. A further ratio with the same method takes seconds, because the calibration and
the factorization are cached.

What to expect:

```
stage 0   saved fine-tuned WavLM (315.45M) -> exp/stage0/wavlm-large-ft.bin
stage 1   [calib] windows per corpus: {'AMI': 150, ...}  total=1050
          [backward] AMI: 600 chunks   ...   [backward] NOTSOFAR: 600 chunks
          [build obdllm-r5] 63.05M (real 5.00x) | ranks L0-11 13048 L12-23 2912 |
                            conv [52, 78, 99, 105, 108, 124]
stage 2   [final] dev_loss=0.3009 | best saved dev_loss=0.3009
```

DER lands in `exp/stage<N>/<tag>/tf_der.txt`, one line per corpus and collar, with the macro on stdout.

A full evaluation is seven corpora and takes hours, so check one first — `eval_tf.sh` rewrites `tf_der.txt`
on every run, so copy it if you want to keep the result:

```bash
SETS="AliMeeting" bash eval_tf.sh exp/stage1/obdllm-r5      # ~15 min, expect DER 15.55
```

Recovery is 17.5 hours; confirm the path works in a few minutes first. `dev_loss` must stay finite — a
diverging run is what the factor rescaling of the paper's Eq. (7) prevents:

```bash
python scripts/recover.py --compressed exp/stage1/obdllm-r5/wavlm-large-ft-obdllm-r5.bin \
    --exp_dir exp/smoke --steps_a 40 --steps_b 40 --epoch_batches 20 --eval_every 20 --dev_subset 2
```

## Results (collar = 0)

DER (%) on seven corpora and their macro-average; Params / MACs are those of the WavLM encoder per second of audio.
Structured pruning is Han et al. (TASLP 2026) reproduced with its released recipe on the same teacher and data.

| System | Params | MACs | AMI | AISHELL-4 | AliMeeting | RAMC | VoxConverse | MSDWild | NOTSOFAR-1 | Macro | Cost | Speed-up GPU / CPU |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| WavLM-Large (teacher) | 315.45M | 17.70G | 14.81 | 10.36 | 14.86 | 10.91 | 9.04 | 16.49 | 21.12 | 13.94 | — | 1.00× / 1.00× |
| Structured pruning 80 % | 63.11M | 3.70G | 14.68 | 9.91 | 13.01 | 11.10 | 8.68 | 16.49 | 21.17 | 13.58 | ≈145 h | 2.76× / 4.28× |
| Low-rank ρ = 2 | 157.62M | 8.49G | 16.14 | 10.82 | 13.67 | 10.78 | 9.80 | 16.62 | 21.26 | 14.16 | 8 min | 1.17× / 1.60× |
| Low-rank ρ = 3 | 105.28M | 5.81G | 17.05 | 11.14 | 14.31 | 12.02 | 10.02 | 17.05 | 21.80 | 14.77 | 8 min | 1.25× / 1.96× |
| Low-rank ρ = 4 | 78.88M | 4.32G | 17.73 | 11.38 | 14.96 | 11.76 | 10.21 | 17.71 | 22.03 | 15.11 | 8 min | 1.30× / 2.22× |
| Low-rank ρ = 5 | 63.11M | 3.47G | 18.10 | 11.39 | 15.30 | 13.66 | 10.21 | 19.20 | 22.43 | 15.76 | 8 min | 1.36× / 2.29× |
| Low-rank ρ = 5 + recovery | 63.11M | 3.47G | 15.71 | 10.58 | 13.78 | 10.49 | 8.99 | 16.86 | 21.40 | **13.97** | + 17.5 h | 1.36× / 2.29× |

- Evaluation follows DiariZen: 16 s inference windows, VBx clustering, one speaker-count range (1–20) and the same
  clustering hyper-parameters for every corpus, no domain adaptation, DER without collar (`dscore`).
- AISHELL-4 is converted to mono; NOTSOFAR-1 uses the single-channel recordings and the official evaluation set.
- Cost is the wall-clock time to obtain the compressed model, excluding the shared teacher: pruning on one H200,
  the low-rank pipeline on one RTX 4090.


## Pretrained models

Three checkpoints are released: the dense fine-tuned teacher, the ρ = 5 low-rank encoder, and the
recovered diarization model (2.0 GB in total).

**Download:** [Google Drive](https://drive.google.com/drive/folders/1AmOPKcasCQal_Sdslg4R9xXHmiwyjuVw?usp=sharing)

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1AmOPKcasCQal_Sdslg4R9xXHmiwyjuVw -O .
```

The download carries its own `pretrained/` and `exp/`, which merge into the repository's when unpacked at
the root. The experiment directories then come ready to score, with no build step:

```bash
md5sum -c checksums.md5
SETS="AliMeeting" bash eval_tf.sh exp/stage2/obdllm-r5_recover_avg5   # ~15 min, expect DER 13.92
bash eval_tf.sh exp/stage2/obdllm-r5_recover_avg5                     # macro DER 13.89
bash eval_tf.sh exp/stage1/obdllm-r5                                  # macro DER 15.76
```

The released low-rank encoder sits in `pretrained/`; a build of your own writes its copy to
`exp/stage1/<tag>/` instead, and either path can be passed to `scripts/recover.py --compressed`.

Scoring still needs the evaluation assets and the `eval` data lists of the *Assets* table; the training and
recovery lists are only needed if you build the models yourself. With the teacher alone, stages 0 and 1
re-derive the compressed encoder in about seven minutes.

Weights are for research and non-commercial use only, see [MODEL_LICENSE](MODEL_LICENSE).

## Citation

The paper is under review; a citation will be added upon publication.

## License

- The **code** is released under the [MIT license](LICENSE). `dataset.py` and `infer_avg.py` are verbatim copies
  from DiariZen (MIT, Brno University of Technology).
- The **model weights** are for research and non-commercial use only, see [MODEL_LICENSE](MODEL_LICENSE).

## Acknowledgments

This work builds on [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) (Han et al.), whose EEND head, powerset
loss, VBx back-end and evaluation protocol are used unchanged; the recovery follows the sequential LoRA of
[SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) and the layer-wise distillation of
[DPHuBERT](https://github.com/pyf98/DPHuBERT). Supported by the KOITA grant funded by MSIT (KOITA 20250002-38).

