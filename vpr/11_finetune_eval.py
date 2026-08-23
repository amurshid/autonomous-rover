#!/usr/bin/env python3
"""Compare the fine-tuning runs on splits with no selection contamination.

`10_finetune.py` writes its own comparison, but each invocation only knows about
its own `--blocks` setting, and its cross-illuminant evaluation queries *both*
held-out night sessions -- session1 and session3 -- against the day-only
database. Session1 is what chose the epoch. Including it makes that headline
number partly a measurement of the thing that was selected on, which is the
mistake the split-by-session discipline exists to prevent.

The margin at stake is small enough for that to matter: the runs land within
0.05 m of each other and of the 09 linear head, and the standing rule here is to
treat anything under 0.02 as noise.

So this reloads each saved backbone, re-encodes every frame with it,
and reports the same evaluations split by query session:

  session3 only   the clean number -- never trained on, never selected on
  session1 only   shown alongside, to expose how much selection bought
  both            what 10_finetune.py printed, for continuity

Nothing is trained here. Run 10_finetune.py for --blocks 4, 8 and 12 first;
this reads out/finetuned_backbone_blocks{N}.pt.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vprlib import data as D          # noqa: E402
from vprlib import features as Fx     # noqa: E402
from vprlib import retrieval as R     # noqa: E402
from vprlib import train as T         # noqa: E402

OUT = HERE / "out"
BLOCKS = (4, 8, 12)
VAL_SESSION = "day1_night_session1"
TEST_SESSION = "day1_night_session3"


def _load_finetune_module():
    """Import 10_finetune.py for its Encoder and encode(). The leading digit
    makes it an invalid identifier, so it cannot be imported by name."""
    spec = importlib.util.spec_from_file_location("ft", HERE / "10_finetune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def encode_cached(ft, df, n_blocks: int, device) -> np.ndarray:
    """Descriptors for every frame under one fine-tuned backbone, cached.

    Same reasoning as features.extract_cached: the weights are fixed by this
    point, so the descriptors are a pure function of (checkpoint, frame list).
    """
    path = OUT / "features" / f"finetuned_blocks{n_blocks}.npy"
    if path.exists():
        print(f"  cached: {path.name}")
        return np.load(path)
    ckpt = OUT / f"finetuned_backbone_blocks{n_blocks}.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt.name} -- run 10_finetune.py --blocks "
                         f"{n_blocks} first")
    print(f"  encoding with {ckpt.name}...")
    model = ft.Encoder(n_blocks).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    # workers=0: macOS starts DataLoader workers with spawn, and a spawned
    # worker re-imports the dataset's module by name. 10_finetune.py was loaded
    # from a path (its leading digit makes it un-importable), so that lookup
    # fails. Single-process loading costs a few minutes once and then caches.
    feats = ft.encode(model, df["path"].tolist(), device, workers=0)
    np.save(path, feats)
    del model
    return feats


def main():
    ft = _load_finetune_module()
    device = ft.pick_device("auto")
    print(f"\n  device: {device}")

    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))

    day_ref = df[df.condition == "day"].reset_index(drop=True)
    val_q = df[df.session == VAL_SESSION].reset_index(drop=True)
    test_q = df[df.session == TEST_SESSION].reset_index(drop=True)
    both_q = df[df.session.isin([VAL_SESSION, TEST_SESSION])].reset_index(drop=True)
    full_ref = df[df.session != TEST_SESSION].reset_index(drop=True)

    print("\n  Loading descriptors:")
    base = Fx.extract_cached(df, backbone="dinov2_vits14")
    variants = [("frozen DINOv2", base)]

    mixed = OUT / "projection_mixed.pt"
    if mixed.exists():
        proj = T.Projection(base.shape[1])
        proj.load_state_dict(torch.load(mixed))
        variants.append(("linear head, day+night (09)", T.apply_projection(proj, base)))

    for n in BLOCKS:
        variants.append((f"fine-tuned, last {n} blocks",
                         encode_cached(ft, df, n, device)))

    # The tuned recipe: 12 blocks, 3x the batch (a 3x larger in-batch negative
    # pool), every anchor each epoch, globally-mined hard negatives and motion
    # blur augmentation. Same split discipline, so it belongs in the same table.
    tuned = OUT / "features" / "finetuned_tuned.npy"
    if tuned.exists():
        variants.append(("fine-tuned, 12 blocks + tuned recipe", np.load(tuned)))

    def score(feats, q_df, ref_df):
        idx, _ = R.retrieve(feats[q_df["_row"].to_numpy()],
                            feats[ref_df["_row"].to_numpy()], k=5)
        return R.evaluate(q_df, ref_df, idx)

    results = {}

    # The clean split first, because it is the one that should be quoted.
    print("\n  === Cross-illuminant, day-only database ===")
    for label, q in (("test session only (clean)", test_q),
                     ("val session only (selected on)", val_q),
                     ("both, as 10_finetune.py reports", both_q)):
        print(f"\n  -- {label}, {len(q)} queries vs {len(day_ref)} day refs")
        for name, f in variants:
            m = score(f, q, day_ref)
            results.setdefault(label, {})[name] = m
            print(R.format_result(name, m))

    print(f"\n  === Full database: {TEST_SESSION} vs the other seven ===")
    for name, f in variants:
        m = score(f, test_q, full_ref)
        results.setdefault("full database, test session", {})[name] = m
        print(R.format_result(name, m))

    # Does fine-tuning help the day side, or only pay for night? evening_700 is
    # in no training set and is the weakest session in the data (README), so it
    # is the honest day-side check.
    ev700 = df[df.session == "day2_evening_700"].reset_index(drop=True)
    ev700_ref = df[~df.session.isin(["day2_evening_700", TEST_SESSION])].reset_index(drop=True)
    print("\n  === Day side: held-out day2_evening_700 ===")
    for name, f in variants:
        m = score(f, ev700, ev700_ref)
        results.setdefault("held-out day session", {})[name] = m
        print(R.format_result(name, m))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, m in results["test session only (clean)"].items():
        e = np.sort(m["_err"])
        axes[0].plot(e, np.arange(1, len(e) + 1) / len(e),
                     label=f"{name} ({m['median_err_m']:.2f} m)")
    axes[0].set_xlim(0, 8)
    axes[0].set_xlabel("localisation error (m)")
    axes[0].set_ylabel("fraction of queries")
    axes[0].set_title(f"Cross-illuminant, {TEST_SESSION} vs day-only database")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # Capacity against accuracy: does unfreezing more of the backbone keep
    # paying, or does 06's overfitting warning show up as a turn-down?
    clean = results["test session only (clean)"]
    xs = list(BLOCKS)
    ys = [clean[f"fine-tuned, last {n} blocks"]["recall@1_1.0m"] for n in xs]
    tuned_key = "fine-tuned, 12 blocks + tuned recipe"
    if tuned_key in clean:
        axes[1].plot([12], [clean[tuned_key]["recall@1_1.0m"]], "s",
                     color="C3", label="+ tuned recipe")
    axes[1].plot(xs, ys, "o-", label="fine-tuned")
    axes[1].axhline(clean["frozen DINOv2"]["recall@1_1.0m"], ls=":", color="0.4",
                    label="frozen DINOv2")
    if "linear head, day+night (09)" in clean:
        axes[1].axhline(clean["linear head, day+night (09)"]["recall@1_1.0m"],
                        ls="--", color="C1", label="linear head (09)")
    axes[1].set_xticks(xs)
    axes[1].set_xlabel("transformer blocks unfrozen (of 12)")
    axes[1].set_ylabel("R@1 <1 m")
    axes[1].set_title("Capacity sweep, clean cross-illuminant split")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "finetune_sweep.png", dpi=110)
    plt.close(fig)

    (OUT / "finetune_sweep.json").write_text(json.dumps(
        {split: {name: {k: v for k, v in m.items() if not k.startswith("_")}
                 for name, m in by_name.items()}
         for split, by_name in results.items()}, indent=2))
    print(f"\n  Wrote finetune_sweep.png/.json to {OUT}")


if __name__ == "__main__":
    main()
