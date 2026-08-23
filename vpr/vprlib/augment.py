"""Photometric night simulation, fitted to this house's actual night statistics.

The first training run failed for an instructive reason: it was validated on
day2_evening_700, whose gap from the other day sessions is "less sunlight",
while the night gap is "tungsten lamps instead of sun". Measured centroid gaps
in DINOv2 space, 0.118 vs 0.197, and the colour casts point in different
directions (R-B of +0.4 by day against +1.7 at night). Correcting the first
shift does not correct the second.

Rather than train on night data -- which would spend the only honest test set
in the project -- this makes the day frames look like night and trains the
projection to be invariant to the change. The parameters below are calibrated
against the real night sessions: darker, warmer, lower contrast, noisier.
"""

from __future__ import annotations

import numpy as np

# Measured from the dataset: day mean 98.4 with R-B +0.4, night mean 86.2 with
# R-B +1.7. The ranges bracket that shift rather than landing exactly on it,
# so the projection sees a spread of lighting rather than one fixed offset.
BRIGHTNESS = (0.75, 1.05)
WARMTH = (1.00, 1.035)     # red gain; blue is attenuated by the same factor
CONTRAST = (0.75, 1.00)
GAMMA = (0.90, 1.20)
NOISE = (0.0, 6.0)         # the Arducam is visibly noisier in low light

# With these values a sample of day frames lands at mean 85.8 / R-B +2.4
# against the real night sessions' 86.2 / +1.7 -- close on brightness, still
# slightly warm, which is the safe direction to err in.


def simulate_night(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """img: HxWx3 uint8 RGB. Returns the same, lit like a house at night."""
    x = img.astype(np.float32) / 255.0

    gamma = rng.uniform(*GAMMA)
    x = np.power(x, gamma)

    warm = rng.uniform(*WARMTH)
    x[..., 0] *= warm
    x[..., 2] /= warm

    x *= rng.uniform(*BRIGHTNESS)

    c = rng.uniform(*CONTRAST)
    x = (x - x.mean()) * c + x.mean()

    sigma = rng.uniform(*NOISE) / 255.0
    if sigma > 0:
        x += rng.normal(0.0, sigma, x.shape).astype(np.float32)

    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


# --- motion blur -----------------------------------------------------------
# Measured on the dataset: variance-of-Laplacian has a median of 167, and the
# blurriest 5% of frames sit below 24. Those frames are ~17x more likely to
# produce a gross retrieval error than a sharp one, so they are worth training
# against rather than only filtering at query time.
#
# The kernel is short because the effect is strong: measured on 200 frames,
# 2 px takes the median from 192 down to ~60, 3 px to 34, and 5 px to 17.5.
# Anything longer produces frames blurrier than the worst real ones, which
# would be training against a failure mode the rover does not have.
BLUR_LEN = (2, 6)


def motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """img: HxWx3 uint8. Linear blur of a random length and direction.

    A rover frame is blurred by the platform turning or driving, which smears
    along one axis -- unlike a Gaussian blur, which smears equally in every
    direction and is the wrong shape for this failure mode.
    """
    import cv2

    length = int(rng.integers(*BLUR_LEN))
    if length < 2:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2),
                                float(rng.uniform(0, 180)), 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    if s <= 0:
        return img
    return cv2.filter2D(img, -1, k / s)
