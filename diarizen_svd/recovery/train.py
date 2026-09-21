#!/usr/bin/env python3
"""One recovery phase (A or B): train the active adapters, evaluate on the dev subset, checkpoint.

Model selection uses the dev loss on a corpus-balanced dev subset; the epoch checkpoints of Phase B
are written to ``<exp_dir>/ep_ckpts/`` and averaged by :mod:`diarizen_svd.recovery.average`.
"""
import os

import torch

from diarizen_svd.recovery.lora import set_phase, save_effective
from diarizen_svd.recovery.losses import dev_loss


def train_phase(model, mods, phase, loader_train, loader_dev, steps, lr, clip, eval_every, exp_dir, best,
                loss_fn, direct_params=(), head_params=(), head_lr=0.0, epoch_batches=0, ep_dir=None,
                writer=None, step_offset=0):
    set_phase(mods, phase)
    lora = [p for m in mods for p in ([m.upA, m.downA] if phase == "A" else [m.upB, m.downB])] + list(direct_params)
    groups = [{"params": lora, "lr": lr}]
    if head_params and head_lr > 0:
        groups.append({"params": list(head_params), "lr": head_lr})
    opt = torch.optim.AdamW(groups)
    params = [p for g in groups for p in g["params"]]
    print(f"[Phase {phase}] {sum(p.numel() for p in lora)/1e6:.3f}M adapter/conv params"
          f"{f' + {sum(p.numel() for p in head_params)/1e6:.2f}M head' if head_params and head_lr > 0 else ''}", flush=True)
    it, step = iter(loader_train), 0
    while step < steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader_train)
            b = next(it)
        xs, t = b["xs"].cuda(), b["ts"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model, xs, t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        step += 1
        if writer is not None and step % 50 == 0:
            writer.add_scalar("Train_Step/Loss", loss.item(), step_offset + step)
        if step % eval_every == 0:
            dl = dev_loss(model, loader_dev)
            if writer is not None:
                writer.add_scalar("Val/Loss", dl, step_offset + step)
            saved = dl < best[0]
            if saved:
                best[0] = dl
                save_effective(model, mods, os.path.join(exp_dir, "checkpoints", "epoch_0001", "pytorch_model.bin"))
            print(f"[Phase {phase}] step {step}: train_loss={loss.item():.4f}  dev_loss={dl:.4f}"
                  f"{'  ** best -> saved' if saved else ''}", flush=True)
        if ep_dir and epoch_batches > 0 and step % epoch_batches == 0:
            ep = step // epoch_batches
            save_effective(model, mods, os.path.join(ep_dir, f"{phase}_ep{ep:02d}.bin"))
            print(f"[Phase {phase}] epoch {ep} checkpoint saved", flush=True)
