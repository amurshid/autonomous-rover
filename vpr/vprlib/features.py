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
    view carries place information, so squash rather than centre-crop."""

    def __init__(self, paths, size=224):
        self.paths = list(paths)
        self.size = size
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std


def build_backbone(name: str):
    """Returns (module, forward_fn, input_size). forward_fn -> (B, D) tensor."""
    if name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        return m, (lambda mod, x: mod(x)), 224

    if name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # CLS token concatenated with mean-pooled patch tokens: the standard
        # DINOv2 retrieval readout, noticeably better than CLS alone.
        def fwd(mod, x):
            out = mod.forward_features(x)
            return torch.cat([out["x_norm_clstoken"],
                              out["x_norm_patchtokens"].mean(1)], dim=1)
        return m, fwd, 224

    raise ValueError(f"unknown backbone: {name}")


@torch.no_grad()
def extract(paths, backbone: str = "resnet50", batch_size: int = 64,
            workers: int = 8, device: str | None = None) -> np.ndarray:
    """L2-normalised descriptors, one row per path."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, fwd, size = build_backbone(backbone)
    model.eval().to(device)

    loader = DataLoader(FrameDataset(paths, size), batch_size=batch_size,
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


def extract_cached(df, backbone: str = "resnet50", **kw) -> np.ndarray:
    """Extract once per (backbone, dataset) and reuse. Keyed on the frame list
    so a changed split or a re-clean invalidates the cache."""
    import hashlib

    key = hashlib.md5("\n".join(df["path"]).encode()).hexdigest()[:12]
    path = CACHE / f"{backbone}_{key}.npy"
    if path.exists():
        print(f"  cached features: {path.name}")
        return np.load(path)
    print(f"  extracting {backbone} for {len(df)} frames...")
    feats = extract(df["path"], backbone=backbone, **kw)
    np.save(path, feats)
    return feats
