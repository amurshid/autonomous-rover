#!/usr/bin/env python3
"""Put night data in the training set and see whether training finally helps.

Everything before this trained on daylight only, and 06 showed why that failed:
fitting to one illuminant narrows the representation onto that illuminant's
cues. The obvious response is to give training both illuminants. Then a mined
positive can pair a *day* frame with a *night* frame at the same spot, which
supervises lighting invariance from real data rather than from the synthetic
approximation in 05.

  train  four day sessions + day1_night_session2      (7,686 day + 1,557 night)
  val    day1_night_session1                          held-out night traversal
  test   day1_night_session3                          never touched

Validating on a held-out night session is the fix for the other half of the
problem in 06. There, validation was a day session and pointed the wrong way;
here it is the same kind of data as the test, so early stopping is aimed at the
thing being measured.

Three models are compared on the same held-out night session, which isolates
exactly what adding night data bought:

  frozen DINOv2                    no training
  projection trained on day only   from 04_metric_learning.py
  projection trained on day+night  this script

WHAT THIS DOES NOT SHOW. The day->night result elsewhere in this repo tests
generalisation to an *unseen illuminant*. That claim is gone the moment night
enters training, and it is not recoverable from this experiment. Worse, the
three night sessions were all recorded on the same evening under the same
lamps, so a held-out night session is a new traversal, not a new lighting
condition. This measures "does mixed-illuminant training help", which is the
deployment question. It does not measure "would this work under lighting never
recorded", and nothing here should be quoted as if it did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def score(feats, ref_df, query_df, k=5):
    idx, _ = R.retrieve(feats[query_df["_row"].to_numpy()],
                        feats[ref_df["_row"].to_numpy()], k=k)
    return R.evaluate(query_df, ref_df, idx)


def main():
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))
    base = Fx.extract_cached(df, backbone="dinov2_vits14")

    train_sessions = DAY_TRAIN + [NIGHT_TRAIN]
    train_df = df[df.session.isin(train_sessions)].reset_index(drop=True)
    val_df = df[df.session == VAL_SESSION].reset_index(drop=True)
    test_df = df[df.session == TEST_SESSION].reset_index(drop=True)
    # Deployment-realistic database: everything except the test session.
    ref_df = df[df.session != TEST_SESSION].reset_index(drop=True)
    # The validation database must exclude the validation session itself --
    # otherwise every query retrieves its own frame, scores 0.00 m, and early
    # stopping has nothing to choose between. Use the training sessions, which
    # exclude both val and test by construction.
    val_ref_df = train_df

    n_night = int((train_df.condition == "night").sum())
    print(f"\n  train {len(train_df)}  ({len(train_df) - n_night} day, "
          f"{n_night} night)")
    print(f"  val   {len(val_df)}  ({VAL_SESSION})")
    print(f"  test  {len(test_df)}  ({TEST_SESSION}), "
          f"database {len(ref_df)}")

    print("\n  Mining positives (different session, <0.5 m, <30 deg yaw)...")
    pos = T.mine_positives(train_df)
    cond = train_df["condition"].to_numpy()
    cross = sum(int((cond[v] != cond[k]).sum()) for k, v in pos.items())
    total = sum(len(v) for v in pos.values())
    print(f"    {len(pos)}/{len(train_df)} anchors usable, {total} pairs")
    print(f"    {cross} of them ({cross / total:.1%}) pair a day frame with a "
          f"night frame\n    -- these are the ones that teach lighting invariance")

    val_curve = []
    best = {"score": -1.0, "epoch": -1, "state": None}

    def on_epoch(epoch, loss, model):
        f = T.apply_projection(model, base)
        m = score(f, val_ref_df, val_df)
        val_curve.append((epoch, loss, m["median_err_m"], m["recall@1_1.0m"]))
        if m["recall@1_1.0m"] > best["score"]:
            best.update(score=m["recall@1_1.0m"], epoch=epoch,
                        state={k: v.clone() for k, v in model.state_dict().items()})
        if epoch % 5 == 0:
            print(f"    epoch {epoch:3d}  loss {loss:.4f}  "
                  f"val median {m['median_err_m']:.2f} m  "
                  f"val R@1<1m {m['recall@1_1.0m']:.3f}")

    print("  Training...")
    model, _ = T.train_projection(base[train_df["_row"].to_numpy()], train_df,
                                  pos, epochs=40, on_epoch=on_epoch)
    print(f"\n  Best epoch {best['epoch']} (val R@1<1m {best['score']:.3f})")
    model.load_state_dict(best["state"])
    feats_mixed = T.apply_projection(model, base)

    variants = [("frozen DINOv2", base)]
    day_only = OUT / "projection.pt"
    if day_only.exists():
        m0 = T.Projection(base.shape[1])
        m0.load_state_dict(torch.load(day_only))
        variants.append(("trained on day only", T.apply_projection(m0, base)))
    variants.append(("trained on day + 1 night session", feats_mixed))

    print(f"\n  === TEST: {TEST_SESSION} vs the other seven sessions ===")
    rows = {}
    for name, f in variants:
        m = score(f, ref_df, test_df)
        rows[name] = m
        print(R.format_result(name, m))

    # Does it still hold up on a day session, or did it just trade one for the
    # other again? evening_700 was never trained on either.
    e700 = df[df.session == "day2_evening_700"].reset_index(drop=True)
    ref_e700 = df[df.session != "day2_evening_700"].reset_index(drop=True)
    print("\n  Sanity check on a held-out DAY session (day2_evening_700):")
    for name, f in variants:
        print(R.format_result(name, score(f, ref_e700, e700)))

    # The sharpest version of the question. Above, the database contains night
    # frames, so nothing has to cross the illuminant gap. Here the database is
    # day-only and the queries are the two night sessions that were never
    # trained on -- so the gap must be crossed, and the only thing that changed
    # between the models is whether training ever saw a night frame.
    day_ref = df[df.condition == "day"].reset_index(drop=True)
    held_night = df[df.session.isin([VAL_SESSION, TEST_SESSION])
                    ].reset_index(drop=True)
    print(f"\n  === Cross-illuminant: {len(held_night)} held-out night queries "
          f"vs a DAY-ONLY database ({len(day_ref)}) ===")
    cross_rows = {}
    for name, f in variants:
        m = score(f, day_ref, held_night)
        cross_rows[name] = m
        print(R.format_result(name, m))

    print("\n  Confidence gate (top-5 spread) on the mixed-trained model:")
    idx, _ = R.retrieve(feats_mixed[test_df["_row"].to_numpy()],
                        feats_mixed[ref_df["_row"].to_numpy()], k=5)
    rxy = ref_df[["x", "y"]].to_numpy()
    txy = test_df[["x", "y"]].to_numpy()
    err = np.hypot(*(rxy[idx[:, 0]] - txy).T)
    topk = rxy[idx]
    spread = np.hypot(*(topk - topk.mean(1, keepdims=True)).transpose(2, 0, 1)).mean(1)
    print(f"  {'gate':>13s} {'kept':>7s} {'median':>9s} {'R@1<1m':>8s} {'err>2m':>8s}")
    for t in (0.25, 0.5, np.inf):
        k = spread <= t
        label = "none" if np.isinf(t) else f"spread<={t}"
        print(f"  {label:>13s} {k.mean():6.1%} {np.median(err[k]):8.2f}m "
              f"{np.mean(err[k] <= 1):8.3f} {np.mean(err[k] > 2):7.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ep, ls, med, rec = zip(*val_curve)
    axes[0].plot(ep, rec, color="tab:orange")
    axes[0].axvline(best["epoch"], ls="--", color="0.5")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val R@1<1m (held-out night session)")
    axes[0].set_title("Validation now tracks the test condition")
    axes[0].grid(alpha=0.3)
    for name, m in rows.items():
        e = np.sort(m["_err"])
        axes[1].plot(e, np.arange(1, len(e) + 1) / len(e),
                     label=f"{name} ({m['median_err_m']:.2f} m)")
    axes[1].set_xlim(0, 8)
    axes[1].set_xlabel("localisation error (m)")
    axes[1].set_ylabel("fraction of queries")
    axes[1].set_title(f"TEST: {TEST_SESSION}")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "mixed_training.png", dpi=110)
    plt.close(fig)

    torch.save(model.state_dict(), OUT / "projection_mixed.pt")
    (OUT / "mixed_training.json").write_text(json.dumps({
        "held_out_night_session_full_db": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in rows.items()},
        "cross_illuminant_day_only_db": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in cross_rows.items()},
    }, indent=2))
    print(f"\n  Wrote mixed_training.png/.json and projection_mixed.pt to {OUT}")


if __name__ == "__main__":
    main()
