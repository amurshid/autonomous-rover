#!/usr/bin/env python3
"""Build the deployment bundle for the Pi.

Everything the rover needs to localise from a camera frame, in one directory,
with nothing that reaches for the network:

  vpr_encoder_traced.pt   TorchScript: architecture + weights in one file, so
                          the Pi needs torch and this file, not torch.hub, not
                          the dinov2 repo, and not an internet connection.
  database.npz            descriptors for all 13,875 frames plus their poses.
  golden.npz              20 frames with their descriptors, as raw JPEG
                          bytes in one buffer plus offsets (never pickled).
  manifest.json           preprocessing constants, checksums, counts.

Two decisions worth explaining.

**The database is re-encoded here, on CPU, through the traced model.** The
descriptors in out/features/ were produced on MPS under bfloat16 autocast, and
the Pi will run CPU float32. That difference is small -- cosine 0.9995 -- and
almost certainly harmless, but the database and the live query should come out
of a numerically identical path so that any future mismatch means a real bug
rather than a known discrepancy the golden test has to tolerate.

**The golden frames travel as JPEG bytes, not paths.** The point of the test is
to prove the Pi's whole pipeline -- decode, channel order, resize, normalise,
model -- reproduces what this machine computed. That only works if the Pi runs
it on the same bytes, and vpr_data/ is not on the Pi.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VPR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPR))
from vprlib import data as D          # noqa: E402
from vprlib import features as Fx     # noqa: E402
from vprlib import retrieval as R     # noqa: E402

OUT = VPR / "out"
BUNDLE = OUT / "bundle"
CHECKPOINT = OUT / "finetuned_backbone_tuned.pt"
BLOCKS = 12
SIZE = 224
N_GOLDEN = 20
GOLDEN_TOL = 0.9999


def load_finetune_module():
    spec = importlib.util.spec_from_file_location("ft", VPR / "10_finetune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preprocess(im: Image.Image) -> torch.Tensor:
    """The one true preprocessing. Must match vprlib.features.FrameDataset and
    whatever the Pi node does, exactly -- the golden test exists to prove it."""
    im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(Fx.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(Fx.IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def encode(model, paths, batch=32):
    out, t0 = [], time.time()
    for i in range(0, len(paths), batch):
        xs = [preprocess(Image.open(p)) for p in paths[i:i + batch]]
        out.append(model(torch.stack(xs)).numpy())
        done = min(i + batch, len(paths))
        print(f"\r    {done}/{len(paths)}  ({time.time() - t0:.0f}s)",
              end="", flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


def main():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists():
        raise SystemExit(f"missing {CHECKPOINT} -- run 10_finetune.py first")

    df = D.label_rooms(D.load_all(verify=False))
    print(f"\n  {len(df)} frames\n")

    print("  Tracing the encoder...")
    ft = load_finetune_module()
    model = ft.Encoder(BLOCKS).cpu()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.randn(1, 3, SIZE, SIZE),
                                 strict=False)
    traced_path = BUNDLE / "vpr_encoder_traced.pt"
    traced.save(traced_path)
    del model, traced

    # Reload the way the Pi will: torch alone, no hub, no source, no network.
    net = torch.jit.load(traced_path)
    net.eval()
    print(f"    {traced_path.name}  "
          f"{traced_path.stat().st_size / 1e6:.1f} MB")

    print("\n  Encoding the database through the traced model (CPU)...")
    feats = encode(net, df["path"].tolist())

    # float16 halves the file and the Pi's RAM for a cosine change of ~1e-4,
    # which is two orders of magnitude below anything that affects retrieval.
    feats16 = feats.astype(np.float16)
    drift = float((feats * feats16.astype(np.float32)).sum(1).mean())
    print(f"    float16 database: cosine to float32 {drift:.6f}")

    np.savez_compressed(
        BUNDLE / "database.npz",
        descriptors=feats16,
        xy=df[["x", "y"]].to_numpy().astype(np.float32),
        yaw=df["yaw"].to_numpy().astype(np.float32),
        session=df["session"].to_numpy().astype("U24"),
        filename=df["filename"].to_numpy().astype("U16"),
        room=df["room"].to_numpy().astype("U24"),
    )

    print("\n  Building the golden test set...")
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(df), N_GOLDEN, replace=False))
    blobs = [Path(p).read_bytes() for p in df.iloc[rows]["path"]]
    # One concatenated uint8 buffer plus offsets, not an object array. An
    # object array can only be serialised by pickling, and a pickle written by
    # numpy 2.x names its internals numpy._core, which the numpy 1.x that ROS
    # Humble's rclpy is built against cannot resolve. The Pi cannot upgrade
    # numpy without breaking rclpy, so the bundle must not depend on pickle.
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    np.savez_compressed(
        BUNDLE / "golden.npz",
        images_blob=np.frombuffer(b"".join(blobs), dtype=np.uint8),
        images_offsets=offsets,
        descriptors=feats[rows],
        names=df.iloc[rows]["filename"].to_numpy().astype("U16"),
    )

    sha = hashlib.sha256(traced_path.read_bytes()).hexdigest()[:16]
    (BUNDLE / "manifest.json").write_text(json.dumps({
        "model": {
            "file": traced_path.name,
            "sha256_16": sha,
            "source_checkpoint": CHECKPOINT.name,
            "blocks_unfrozen": BLOCKS,
            "descriptor_dim": int(feats.shape[1]),
        },
        "preprocessing": {
            "size": SIZE,
            "resample": "PIL bilinear",
            "channel_order": "RGB (flip msg.data if encoding is bgr8)",
            "scale": "uint8 / 255",
            "mean": list(Fx.IMAGENET_MEAN),
            "std": list(Fx.IMAGENET_STD),
        },
        "database": {
            "frames": int(len(df)),
            "dtype": "float16",
            "sessions": sorted(df["session"].unique().tolist()),
        },
        "retrieval": {
            "k": 5,
            "aggregation_temp": R.AGG_TEMP,
            "sequence_n": R.SEQ_N,
            "sequence_base_m": R.SEQ_BASE,
            "frame_spacing_m": R.FRAME_SPACING,
            "gate_spread_m": 0.5,
        },
        "golden": {"n": N_GOLDEN, "tolerance": GOLDEN_TOL},
    }, indent=2))

    print("\n  Verifying: re-running the golden frames through the bundle...")
    got = np.stack([net(preprocess(Image.open(__import__("io").BytesIO(b)))[None])
                    .detach().numpy()[0] for b in blobs])
    cos = (got * feats[rows]).sum(1)
    print(f"    cosine mean {cos.mean():.6f}  min {cos.min():.6f}")
    if cos.min() < GOLDEN_TOL:
        raise SystemExit("golden verification FAILED -- bundle is inconsistent")

    # And end to end: does the bundle actually localise?
    idx, sim = R.retrieve(feats[rows], feats, k=6)
    idx, sim = idx[:, 1:], sim[:, 1:]        # drop self-match
    xy = df[["x", "y"]].to_numpy()
    err = np.hypot(*(R.aggregate(xy, idx, sim) - xy[rows]).T)
    print(f"    end-to-end on the golden frames: median {np.median(err):.2f} m")

    total = sum(f.stat().st_size for f in BUNDLE.iterdir())
    print(f"\n  Bundle at {BUNDLE}  ({total / 1e6:.0f} MB)")
    for f in sorted(BUNDLE.iterdir()):
        print(f"    {f.name:26s} {f.stat().st_size / 1e6:7.1f} MB")
    print("\n  Copy this directory to the Pi and run scripts/vpr_relocalise.py")


if __name__ == "__main__":
    main()
