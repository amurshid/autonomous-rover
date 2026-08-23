#!/usr/bin/env python3
"""Step 3 of the progression, properly: fine-tune the backbone.

Everything up to 09 trained a linear projection on top of frozen descriptors.
That can only re-weight the 768 numbers DINOv2 already produced -- it cannot
recover a cue the backbone discarded. Fine-tuning can, which is the reason to
try it.

The regime matters. 06 showed training hurt when the training set held one
illuminant, and 09 showed the same code helped once night data was mixed in. So
this runs in the 09 regime, not the 04 one:

  train  four day sessions + day1_night_session2   (7,686 day + 1,557 night)
  val    day1_night_session1 vs the training sessions
  test   day1_night_session3 vs the other seven    (touched once)

Training setup, and why each piece is there:

  --blocks N    unfreeze the last N transformer blocks. Early blocks hold
                generic edge and texture features that do not need adjusting.
  --llrd        layer-wise learning rate decay. The graded version of freezing:
                each block trains at a fraction of the rate of the one after it,
                so the whole backbone can stay trainable without early layers
                being dragged off their pretrained solution. Makes --blocks 12
                a reasonable thing to run rather than a reckless one.
  --warmup      linear warmup then cosine decay. The first optimiser steps are
                where a pretrained backbone gets damaged, because AdamW's
                second-moment estimate starts near zero and the effective step
                is much larger than the nominal rate.
  --amp         bfloat16 autocast, roughly 2x throughput. The loss itself stays
                in fp32 -- cosine similarities are too closely spaced for bf16's
                mantissa to rank reliably.
  --batch-pairs not only a memory knob. The loss mines its hardest negative from
                inside the batch, so a larger batch is a larger negative pool
                and a harder objective. Gradient accumulation does not
                substitute here: it grows the gradient, not the negative pool.

Anchors are subsampled per epoch. Gradients are clipped at norm 1.0.

Descriptors cannot be cached here -- the backbone changes every step, so images
are loaded and encoded every epoch. That is the whole reason this is slower than
everything before it.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vprlib import augment as Aug     # noqa: E402
from vprlib import data as D          # noqa: E402
from vprlib import features as Fx     # noqa: E402
from vprlib import retrieval as R     # noqa: E402
from vprlib import train as T         # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
DAY_TRAIN = ["day2_afternoon_300", "day2_afternoon_400",
             "day2_evening_500", "day2_evening_600"]
NIGHT_TRAIN = "day1_night_session2"
VAL_SESSION = "day1_night_session1"
TEST_SESSION = "day1_night_session3"


def load_img(path, size=224, blur_rng=None):
    im = Image.open(path).convert("RGB")
    if blur_rng is not None:
        # Blur at native 320x240, before the resize -- the same order the
        # camera and the resize pipeline would apply it in.
        im = Image.fromarray(Aug.motion_blur(np.array(im), blur_rng))
    im = im.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(Fx.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(Fx.IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


class PairDataset(Dataset):
    """Yields (anchor, positive) image pairs and their poses.

    With `neg_index`, each item also carries a globally-mined hard negative --
    a frame that looks like the anchor to the current descriptor but was
    recorded far away. Those are the corridor confusions that produce the error
    tail, and sampling them deliberately puts them in front of the loss instead
    of waiting for one to turn up in a random batch.
    """

    def __init__(self, df, anchors, pos_index, seed=0, neg_index=None,
                 blur_p=0.0):
        self.paths = df["path"].to_numpy()
        self.xy = df[["x", "y"]].to_numpy().astype(np.float32)
        self.anchors = anchors
        self.pos_index = pos_index
        self.neg_index = neg_index
        self.blur_p = blur_p
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.anchors)

    def _load(self, row, rng):
        # Per-item, per-epoch seed: DataLoader workers each get a copy of the
        # dataset, so a shared rng would repeat across workers.
        blur = rng if (self.blur_p and rng.random() < self.blur_p) else None
        return load_img(self.paths[row], blur_rng=blur)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        a = int(self.anchors[i])
        p = int(rng.choice(self.pos_index[a]))
        out = [self._load(a, rng), self._load(p, rng),
               torch.from_numpy(self.xy[a]), torch.from_numpy(self.xy[p])]
        if self.neg_index is not None:
            n = int(rng.choice(self.neg_index[a]))
            out += [self._load(n, rng), torch.from_numpy(self.xy[n])]
        return tuple(out)


def mine_hard_negatives(feats, xy, anchors, min_dist=5.0, top=20, block=512):
    """For each anchor, the most similar frames that are recorded far away.

    These are perceptual aliases -- two stretches of corridor that genuinely
    look alike -- and they are what the error tail is made of: 4% of queries
    carry 40% of the total error mass. The in-batch mining in
    batch_hard_triplet can only pick the hardest negative that happened to be
    sampled; this finds the hardest ones in the whole training set.

    Mined from the frozen descriptors so the pool does not depend on which runs
    happened before it.
    """
    out = {}
    anchors = np.asarray(anchors)
    k = min(top, len(feats) - 1)
    for s in range(0, len(anchors), block):
        rows = anchors[s:s + block]
        sim = feats[rows] @ feats.T
        d = np.hypot(*(xy[None, :, :] - xy[rows][:, None, :]).transpose(2, 0, 1))
        sim[d <= min_dist] = -2.0
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        for j, r in enumerate(rows):
            cand = idx[j][sim[j, idx[j]] > -2.0]
            if len(cand):
                out[int(r)] = cand
    return out


class Encoder(torch.nn.Module):
    """DINOv2 with only the last `n_blocks` transformer blocks trainable.

    Early blocks hold generic low-level features -- edges, texture, colour
    gradients -- which are as correct for this house as for anything else, and
    are the expensive half of the backward pass. Late blocks carry the semantic
    representation, which is what adapting to a specific place actually needs.

    Capacity matters more than speed here. 06 showed 590k parameters were enough
    to overfit onto daylight cues on this dataset; the full backbone is 22M.
    Pass --blocks 12 to unfreeze everything and test that directly.
    """

    def __init__(self, n_blocks=4):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2",
                                       "dinov2_vits14", verbose=False)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.n_depth = len(self.backbone.blocks)
        n_blocks = min(n_blocks, self.n_depth)
        self.n_blocks = n_blocks
        trainable = list(self.backbone.blocks[-n_blocks:]) + [self.backbone.norm]
        if n_blocks >= self.n_depth:
            trainable.append(self.backbone.patch_embed)
        for m in trainable:
            for p in m.parameters():
                p.requires_grad = True
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"    trainable {n_tr / 1e6:.1f}M of {n_all / 1e6:.1f}M parameters "
              f"(last {n_blocks} of {self.n_depth} blocks)")

    def param_groups(self, base_lr: float, decay: float):
        """Layer-wise learning rate decay.

        Block depth maps to how task-specific a layer is: the last block carries
        the semantic representation and should move, the first carries edges and
        texture and should barely move. Freezing is the crude version of this --
        a hard zero for early layers. Scaling the rate by depth instead lets the
        whole backbone stay trainable without the early layers being dragged off
        their pretrained solution, which is what --blocks 12 needs to be safe.

        block d gets base_lr * decay ** (depth - 1 - d), so the final block runs
        at base_lr and each earlier one at a fraction of it.
        """
        groups, seen = [], set()

        def add(module, lr, tag):
            ps = [p for p in module.parameters() if p.requires_grad
                  and id(p) not in seen]
            if not ps:
                return
            seen.update(id(p) for p in ps)
            groups.append({"params": ps, "lr": lr, "tag": tag})

        add(self.backbone.norm, base_lr, "norm")
        for d in range(self.n_depth - 1, -1, -1):
            scale = decay ** (self.n_depth - 1 - d)
            add(self.backbone.blocks[d], base_lr * scale, f"block{d}")
        add(self.backbone.patch_embed, base_lr * decay ** self.n_depth, "patch")

        if groups:
            lrs = [g["lr"] for g in groups]
            print(f"    layer-wise LR decay {decay}: "
                  f"{max(lrs):.2e} (last) down to {min(lrs):.2e} (first), "
                  f"{len(groups)} groups")
        return groups

    def forward(self, x):
        out = self.backbone.forward_features(x)
        f = torch.cat([out["x_norm_clstoken"],
                       out["x_norm_patchtokens"].mean(1)], dim=1)
        return F.normalize(f, dim=1)


class PathDataset(Dataset):
    """Plain image loader. Module-level on purpose: Windows spawns DataLoader
    workers and cannot pickle a class defined inside a function."""

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_img(self.paths[i])


def warmup_cosine(step: int, total: int, warmup_frac: float = 0.1) -> float:
    """LR multiplier: linear warmup, then cosine decay toward zero.

    Warmup exists because the first optimiser steps are the dangerous ones.
    AdamW's second-moment estimate starts near zero, so the effective step size
    early on is far larger than the nominal rate -- which is exactly when a
    pretrained backbone gets damaged. Ramping in avoids that. The cosine tail
    then lets training settle instead of bouncing at full rate to the last step.
    """
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def amp_context(device: torch.device, enabled: bool):
    """bfloat16 autocast where it is supported, a no-op otherwise.

    bf16 rather than fp16 because it keeps float32's exponent range, so no
    GradScaler is needed -- and GradScaler is CUDA-only anyway, which would not
    help on Apple Silicon.
    """
    if not enabled or device.type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def pick_device(requested: str = "auto") -> torch.device:
    """cuda > mps > cpu. MPS is the Apple Silicon GPU backend."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode(model, paths, device, batch_size=64, workers=6, amp=True):
    """Descriptors for a list of paths using the current weights."""
    model.eval()
    loader = DataLoader(PathDataset(paths), batch_size=batch_size,
                        num_workers=workers)
    out = []
    for b in loader:
        with amp_context(device, amp):
            f = model(b.to(device))
        out.append(f.float().cpu().numpy())
    model.train()
    return np.concatenate(out).astype(np.float32)


def score_from(feats_ref, feats_q, ref_df, q_df):
    idx, _ = R.retrieve(feats_q, feats_ref, k=5)
    return R.evaluate(q_df, ref_df, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=4,
                    help="how many trailing transformer blocks to unfreeze")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--anchors-per-epoch", type=int, default=3000)
    # Batch size is not just a memory knob here. batch_hard_triplet mines its
    # hardest negative from inside the batch, so a bigger batch means a bigger
    # negative pool and a harder, more informative loss. Gradient accumulation
    # would NOT substitute: it enlarges the gradient but each micro-batch still
    # only sees its own 16 pairs, so the negatives stay just as easy.
    ap.add_argument("--batch-pairs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--llrd", type=float, default=0.75,
                    help="layer-wise LR decay per block, 1.0 disables it")
    ap.add_argument("--warmup", type=float, default=0.1,
                    help="fraction of total steps spent warming up")
    ap.add_argument("--amp", action="store_true", default=True,
                    help="bfloat16 autocast (default on; --no-amp to disable)")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--hard-negatives", type=int, default=0,
                    help="pool size per anchor for globally-mined hard "
                         "negatives; 0 keeps in-batch mining only")
    ap.add_argument("--neg-min-dist", type=float, default=5.0,
                    help="a mined negative must be at least this far away")
    ap.add_argument("--blur-p", type=float, default=0.0,
                    help="probability of motion-blurring a training image")
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--val-stride", type=int, default=4,
                    help="subsample the validation reference set by this factor")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | mps  (mps = Apple Silicon GPU)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    print(f"\n  device: {device}")
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))

    train_df = df[df.session.isin(DAY_TRAIN + [NIGHT_TRAIN])].reset_index(drop=True)
    val_df = df[df.session == VAL_SESSION].reset_index(drop=True)
    test_df = df[df.session == TEST_SESSION].reset_index(drop=True)
    ref_df = df[df.session != TEST_SESSION].reset_index(drop=True)
    val_ref = train_df.iloc[::args.val_stride].reset_index(drop=True)

    print(f"\n  train {len(train_df)}  val {len(val_df)} vs {len(val_ref)}  "
          f"test {len(test_df)} vs {len(ref_df)}")

    print("\n  Mining positives...")
    pos = T.mine_positives(train_df)
    keys = np.array(list(pos))
    cond = train_df["condition"].to_numpy()
    cross = sum(int((cond[v] != cond[k]).sum()) for k, v in pos.items())
    total = sum(len(v) for v in pos.values())
    print(f"    {len(pos)} anchors, {cross / total:.1%} of pairs cross day/night")

    neg = None
    if args.hard_negatives:
        print("\n  Mining hard negatives from the frozen descriptors...")
        base_train = Fx.extract_cached(df, backbone="dinov2_vits14")[
            train_df["_row"].to_numpy()]
        neg = mine_hard_negatives(base_train, train_df[["x", "y"]].to_numpy(),
                                  list(pos), min_dist=args.neg_min_dist,
                                  top=args.hard_negatives)
        d = [float(np.hypot(*(train_df[["x", "y"]].to_numpy()[v]
                              - train_df[["x", "y"]].to_numpy()[k]).T).mean())
             for k, v in list(neg.items())[:500]]
        print(f"    {len(neg)} anchors have a pool, mean distance to their "
              f"mined negatives {np.mean(d):.1f} m")
        keys = np.array([k for k in keys if k in neg])

    print("\n  Building model:")
    model = Encoder(args.blocks).to(device)
    model.train()
    if args.llrd < 1.0:
        groups = model.param_groups(args.lr, args.llrd)
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": args.lr, "tag": "all"}]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = max(1, min(args.anchors_per_epoch, len(keys)) // args.batch_pairs)
    total_steps = steps_per_epoch * args.epochs
    print(f"    {steps_per_epoch} steps/epoch, {total_steps} total, "
          f"amp={'bf16' if args.amp and device.type != 'cpu' else 'off'}")
    step = 0

    val_paths = val_df["path"].tolist()
    val_ref_paths = val_ref["path"].tolist()

    def validate():
        f_ref = encode(model, val_ref_paths, device, workers=args.workers, amp=args.amp)
        f_q = encode(model, val_paths, device, workers=args.workers, amp=args.amp)
        return score_from(f_ref, f_q, val_ref, val_df)

    print("\n  Validating before training (should match the frozen baseline)...")
    m0 = validate()
    print(R.format_result("  epoch -1 (frozen)", m0))
    best = {"score": m0["recall@1_1.0m"], "epoch": -1,
            "state": copy.deepcopy(model.state_dict())}
    curve = [(-1, float("nan"), m0["recall@1_1.0m"])]

    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        t0 = time.time()
        sel = rng.choice(keys, size=min(args.anchors_per_epoch, len(keys)),
                         replace=False)
        ds = PairDataset(train_df, sel, pos, seed=epoch + 1, neg_index=neg,
                         blur_p=args.blur_p)
        loader = DataLoader(ds, batch_size=args.batch_pairs, shuffle=True,
                            num_workers=args.workers, drop_last=True)

        losses = []
        for batch in loader:
            if neg is None:
                anc_img, pos_img, anc_xy, pos_xy = batch
                neg_img = neg_xy = None
            else:
                anc_img, pos_img, anc_xy, pos_xy, neg_img, neg_xy = batch
            mult = warmup_cosine(step, total_steps, args.warmup)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * mult

            parts = [anc_img, pos_img] + ([neg_img] if neg_img is not None else [])
            imgs = torch.cat(parts, dim=0).to(device)
            with amp_context(device, args.amp):
                emb_all = model(imgs)
            # The loss runs in fp32: cosine similarities sit close together and
            # bf16's 8-bit mantissa is too coarse to rank them reliably.
            emb_all = emb_all.float()

            b = len(anc_img)
            # Interleave so rows 2i / 2i+1 are a pair, as batch_hard_triplet
            # expects. Built with stack/reshape to keep the autograd graph.
            emb = torch.stack([emb_all[:b], emb_all[b:2 * b]],
                              dim=1).reshape(2 * b, -1)
            xy = torch.stack([anc_xy, pos_xy], dim=1).reshape(2 * b, 2).to(device)
            if neg_img is not None:
                # Appended after the pair rows: scored as candidate negatives,
                # never used as an anchor or a positive.
                emb = torch.cat([emb, emb_all[2 * b:]], dim=0)
                xy = torch.cat([xy, neg_xy.to(device)], dim=0)

            loss = T.batch_hard_triplet(emb, xy, margin=args.margin,
                                        n_pairs=b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1

        mins = (time.time() - t0) / 60
        line = (f"    epoch {epoch:3d}  loss {np.mean(losses):.4f}  "
                f"lr {opt.param_groups[0]['lr']:.2e}  ({mins:.1f} min)")
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            m = validate()
            curve.append((epoch, float(np.mean(losses)), m["recall@1_1.0m"]))
            line += (f"  val median {m['median_err_m']:.2f} m  "
                     f"R@1<1m {m['recall@1_1.0m']:.3f}")
            if m["recall@1_1.0m"] > best["score"]:
                best.update(score=m["recall@1_1.0m"], epoch=epoch,
                            state=copy.deepcopy(model.state_dict()))
                line += "  *"
        print(line, flush=True)

    print(f"\n  Best epoch {best['epoch']} (val R@1<1m {best['score']:.3f})")
    model.load_state_dict(best["state"])

    print("\n  Encoding everything with the fine-tuned backbone...")
    feats_ft = encode(model, df["path"].tolist(), device, workers=args.workers,
                      amp=args.amp)
    np.save(OUT / "features" / "finetuned_full.npy", feats_ft)
    base = Fx.extract_cached(df, backbone="dinov2_vits14")

    def rows_for(f):
        return score_from(f[ref_df["_row"].to_numpy()],
                          f[test_df["_row"].to_numpy()], ref_df, test_df)

    variants = [("frozen DINOv2", base)]
    mixed = OUT / "projection_mixed.pt"
    if mixed.exists():
        m1 = T.Projection(base.shape[1])
        m1.load_state_dict(torch.load(mixed))
        variants.append(("linear head, day+night (09)", T.apply_projection(m1, base)))
    variants.append((f"fine-tuned last {args.blocks} blocks", feats_ft))

    print(f"\n  === TEST: {TEST_SESSION} vs the other seven ===")
    rows = {}
    for name, f in variants:
        m = rows_for(f)
        rows[name] = m
        print(R.format_result(name, m))

    day_ref = df[df.condition == "day"].reset_index(drop=True)
    held = df[df.session.isin([VAL_SESSION, TEST_SESSION])].reset_index(drop=True)
    print("\n  === Cross-illuminant: held-out night vs DAY-ONLY database ===")
    cross_rows = {}
    for name, f in variants:
        m = score_from(f[day_ref["_row"].to_numpy()],
                       f[held["_row"].to_numpy()], day_ref, held)
        cross_rows[name] = m
        print(R.format_result(name, m))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ep, ls, rec = zip(*curve)
    axes[0].plot(ep, rec, "o-")
    axes[0].axvline(best["epoch"], ls="--", color="0.5")
    axes[0].set_xlabel("epoch (-1 = frozen)")
    axes[0].set_ylabel("val R@1<1m")
    axes[0].set_title("Fine-tuning")
    axes[0].grid(alpha=0.3)
    for name, m in cross_rows.items():
        e = np.sort(m["_err"])
        axes[1].plot(e, np.arange(1, len(e) + 1) / len(e),
                     label=f"{name} ({m['median_err_m']:.2f} m)")
    axes[1].set_xlim(0, 8)
    axes[1].set_xlabel("localisation error (m)")
    axes[1].set_title("Cross-illuminant, day-only database")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "finetune.png", dpi=110)
    plt.close(fig)

    torch.save(best["state"], OUT / "finetuned_backbone.pt")
    (OUT / "finetune.json").write_text(json.dumps({
        "args": vars(args),
        "best_epoch": best["epoch"],
        "held_out_night_full_db": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in rows.items()},
        "cross_illuminant_day_only_db": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in cross_rows.items()},
    }, indent=2))
    print(f"\n  Wrote finetune.png/.json and finetuned_backbone.pt to {OUT}")


if __name__ == "__main__":
    main()
