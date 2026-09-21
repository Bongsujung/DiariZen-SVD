#!/usr/bin/env python3
"""Recovery fine-tuning of a low-rank WavLM diarization model (paper §2.6, Eq. 8).

Command-line front end; the pieces live in :mod:`diarizen_svd.recovery` --
``lora`` (sequential LoRA on the factor pair), ``losses`` (Eq. 8 and the dev loss) and
``train`` (one phase).  Only the factor adapters, the CNN factor pairs and -- optionally -- the EEND
head are trained; everything else is frozen.

Paper setting (rho = 5):
  python scripts/recover.py --compressed exp/stage1/obdllm-r5/wavlm-large-ft-obdllm-r5.bin --exp_dir exp/stage2/obdllm-r5_recover \
      --lora_rank 128 --lr 5e-5 --head_lr 2e-5 --epochs_a 5 --epochs_b 5 --batch 16 \
      --distill_lambda 1.0 --distill_S 0,2,4,6,8,16,24 --dev_subset 10 --eval_every 2000
"""
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diarizen_svd.paths import DEFAULT, set_seed
from diarizen_svd.experiment import HEAD_LARGE, write_exp_scaffold
from diarizen_svd.data.corpus import balanced_dev_subset
from diarizen_svd.data.loader import make_loader
from diarizen_svd.nn.model import Model
from diarizen_svd.recovery.lora import inject_lora, merge, save_effective
from diarizen_svd.recovery.losses import FeatureTap, recovery_loss, dev_loss
from diarizen_svd.recovery.train import train_phase


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compressed", required=True, help="low-rank WavLM checkpoint (stage 1 output, exp/stage1/<tag>/)")
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--ft_diar", default=DEFAULT["teacher_diar"], help="EEND checkpoint providing the trained head")
    ap.add_argument("--teacher", default=DEFAULT["teacher_wavlm"], help="dense fine-tuned WavLM for feature distillation")
    ap.add_argument("--train_dir", default=DEFAULT["recovery_train"], help="recovery lists: {wav.scp, rttm, all.uem} (data/_pooled/recover)")
    ap.add_argument("--dev_dir", default=DEFAULT["recovery_dev"], help="validation lists: {wav.scp, rttm, all.uem} (data/_pooled/dev)")
    ap.add_argument("--mode", choices=["task", "distill", "both"], default="both")
    ap.add_argument("--distill_lambda", type=float, default=1.0)
    ap.add_argument("--distill_S", default="0,2,4,6,8,16,24", help="WavLM layers matched (0 = CNN output)")
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5, help="adapters and conv factors")
    ap.add_argument("--head_lr", type=float, default=2e-5, help="> 0: also fine-tune the EEND head at this lr")
    ap.add_argument("--epochs_a", type=float, default=5)
    ap.add_argument("--epochs_b", type=float, default=5)
    ap.add_argument("--steps_a", type=int, default=0, help="overrides --epochs_a when > 0")
    ap.add_argument("--steps_b", type=int, default=0, help="overrides --epochs_b when > 0")
    ap.add_argument("--epoch_batches", type=int, default=0, help="batches per 'epoch' checkpoint (0 = one pass over the data)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--chunk_shift", type=int, default=6)
    ap.add_argument("--dev_chunk_size", type=int, default=8)
    ap.add_argument("--dev_subset", type=int, default=10, help="recordings per corpus in the dev subset (0 = full dev)")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--head", default="large", choices=["large"], help="EEND head preset (HEAD_LARGE)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    head = HEAD_LARGE

    # ---- student = compressed WavLM + teacher head; only the factor adapters (+ conv factors, + head) train
    m = Model(wavlm_src=args.compressed, **head)
    ft = torch.load(args.ft_diar, map_location="cpu")
    ft = ft.get("state_dict", ft) if isinstance(ft, dict) else ft
    res = m.load_state_dict({k: v for k, v in ft.items() if not k.startswith("wavlm_model.")}, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys[:5]
    m.cuda()
    for p in m.parameters():
        p.requires_grad_(False)
    conv_ps = [p for n, p in m.named_parameters() if ".conv_A." in n or ".conv_B." in n]
    for p in conv_ps:
        p.requires_grad_(True)
    print(f"[conv] {sum(p.numel() for p in conv_ps)/1e3:.0f}K CNN factor params trained directly", flush=True)
    head_ps = []
    if args.head_lr > 0:
        head_ps = [p for n, p in m.named_parameters() if not n.startswith("wavlm_model.")]
        for p in head_ps:
            p.requires_grad_(True)
    mods = inject_lora(m, args.lora_rank)
    m.train()

    # ---- data
    tr, dv = args.train_dir, args.dev_dir
    Ltr = make_loader(m, f"{tr}/wav.scp", f"{tr}/rttm", f"{tr}/all.uem", args.batch,
                      args.chunk_size, args.chunk_shift, shuffle=True, num_workers=2)
    epb = args.epoch_batches or len(Ltr)
    steps_a = args.steps_a or int(args.epochs_a * epb)
    steps_b = args.steps_b or int(args.epochs_b * epb)
    print(f"[data] {len(Ltr)} batches/epoch -> steps_a={steps_a} steps_b={steps_b} (checkpoint every {epb} batches)", flush=True)
    if args.dev_subset > 0:
        dscp, drttm, duem = balanced_dev_subset(dv, args.dev_subset, args.exp_dir, args.seed)
    else:
        dscp, drttm, duem = f"{dv}/wav.scp", f"{dv}/rttm", f"{dv}/all.uem"
    Ldev = make_loader(m, dscp, drttm, duem, args.batch, args.dev_chunk_size, args.dev_chunk_size, shuffle=False, num_workers=2)
    write_exp_scaffold(args.exp_dir, args.compressed, head)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, "tb"))
    ep_dir = os.path.join(args.exp_dir, "ep_ckpts")

    # ---- objective
    teacher = None
    S = tuple(int(x) for x in args.distill_S.split(","))
    if args.mode != "task":
        teacher = Model(wavlm_src=args.teacher, **head).cuda().eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    tap = FeatureTap(m)
    loss_fn = lambda mm, xs, t: recovery_loss(mm, teacher, xs, t, S, args.distill_lambda, args.mode, tap)
    print(f"[mode={args.mode} lambda={args.distill_lambda} S={S}]", flush=True)

    best = [float("inf")]
    init = dev_loss(m, Ldev)
    best[0] = init
    save_effective(m, mods, os.path.join(args.exp_dir, "checkpoints", "epoch_0001", "pytorch_model.bin"))
    print(f"[init] dev_loss={init:.4f} (truncation-free model saved as baseline)", flush=True)

    common = dict(loader_train=Ltr, loader_dev=Ldev, lr=args.lr, clip=args.clip, eval_every=args.eval_every,
                  exp_dir=args.exp_dir, best=best, loss_fn=loss_fn, direct_params=conv_ps, head_params=head_ps,
                  head_lr=args.head_lr, epoch_batches=epb, ep_dir=ep_dir, writer=writer)
    train_phase(m, mods, "A", steps=steps_a, step_offset=0, **common)
    merge(mods, "A")
    train_phase(m, mods, "B", steps=steps_b, step_offset=steps_a, **common)
    merge(mods, "B")
    for mm in mods:
        mm.act_A = mm.act_B = False
    fin = dev_loss(m, Ldev)
    if fin < best[0]:
        best[0] = fin
        save_effective(m, mods, os.path.join(args.exp_dir, "checkpoints", "epoch_0001", "pytorch_model.bin"))
    print(f"[final] dev_loss={fin:.4f} | best saved dev_loss={best[0]:.4f}", flush=True)
    print(f"best -> {args.exp_dir}/checkpoints/epoch_0001/pytorch_model.bin ; epoch checkpoints -> {ep_dir}")


if __name__ == "__main__":
    main()
