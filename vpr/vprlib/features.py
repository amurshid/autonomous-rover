"""Frozen-backbone feature extraction.

No training here — an ImageNet/DINO backbone is used purely as a feature
extractor, which is the second step of the progression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CACHE = Path(__file__).resolve().parents[1] / "out" / "features"
CACHE.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameDataset(Dataset):
    """Frames resized to a square. The source is 320x240 and the whole field of
    view carries place information, so squash rather than centre-crop.

    `augment` is an optional callable (img_uint8, rng) -> img_uint8, applied
    before the resize. `seed` makes the augmentation reproducible across the
    DataLoader's worker processes."""

    def __init__(self, paths, size=224, augment=None, seed=0):
        self.paths = list(paths)
        self.size = size
        self.augment = augment
        self.seed = seed
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.augment is not None:
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            im = Image.fromarray(self.augment(np.array(im), rng))
        # size may be an int (square) or (width, height); ViT needs both
        # dimensions to be multiples of the 14-pixel patch.
        wh = (self.size, self.size) if isinstance(self.size, int) else self.size
        im = im.resize(wh, Image.BILINEAR)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std


def gem(x, p=3.0, eps=1e-6):
    """Generalised mean over patch tokens. p=1 is average pooling, p->inf is
    max. Intermediate p emphasises the strongest-responding patches, which for
    retrieval means distinctive objects rather than the floor filling most of
    the frame."""
    return x.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


def build_backbone(name: str, pooling: str = "cls+mean"):
    """Returns (module, forward_fn, default_size). forward_fn -> (B, D)."""
    if name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        return m, (lambda mod, x: mod(x)), 224

    if name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           verbose=False)

        def fwd(mod, x):
            out = mod.forward_features(x)
            cls, patches = out["x_norm_clstoken"], out["x_norm_patchtokens"]
            if pooling == "cls":
                return cls
            if pooling == "mean":
                return patches.mean(1)
            if pooling == "gem":
                return gem(patches)
            if pooling == "cls+mean":
                return torch.cat([cls, patches.mean(1)], dim=1)
            if pooling == "cls+gem":
                return torch.cat([cls, gem(patches)], dim=1)
            raise ValueError(f"unknown pooling: {pooling}")

        return m, fwd, 224

    raise ValueError(f"unknown backbone: {name}")


@torch.no_grad()
def extract(paths, backbone: str = "resnet50", batch_size: int = 64,
            workers: int = 8, device: str | None = None,
            augment=None, seed: int = 0, pooling: str = "cls+mean",
            size=None) -> np.ndarray:
    """L2-normalised descriptors, one row per path."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, fwd, default_size = build_backbone(backbone, pooling=pooling)
    model.eval().to(device)

    loader = DataLoader(FrameDataset(paths, size or default_size,
                                     augment=augment, seed=seed),
                        batch_size=batch_size,
                        num_workers=workers, shuffle=False, pin_memory=False)

    chunks = []
    done = 0
    for batch in loader:
        feats = fwd(model, batch.to(device))
        chunks.append(F.normalize(feats.float(), dim=1).cpu().numpy())
        done += len(batch)
        print(f"\r    {done}/{len(loader.dataset)}", end="", flush=True)
    print()
    return np.concatenate(chunks).astype(np.float32)


def extract_cached(df, backbone: str = "resnet50", tag: str = "", **kw) -> np.ndarray:
    """Extract once per (backbone, dataset) and reuse. Keyed on the frame list
    so a changed split or a re-clean invalidates the cache. `tag` separates
    augmented passes over the same frames."""
    import hashlib

    key = hashlib.md5("\n".join(df["path"]).encode()).hexdigest()[:12]
    suffix = f"_{tag}" if tag else ""
    path = CACHE / f"{backbone}_{key}{suffix}.npy"
    if path.exists():
        print(f"  cached features: {path.name}")
        return np.load(path)
    print(f"  extracting {backbone} for {len(df)} frames...")
    feats = extract(df["path"], backbone=backbone, **kw)
    np.save(path, feats)
    return feats
