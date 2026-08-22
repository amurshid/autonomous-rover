# VPR — retrieval baseline

No training yet: a frozen
backbone is used as a feature extractor and places are recognised by nearest-
neighbour retrieval against a reference set.

```
python 01_explore.py                          # load, verify, trajectories, BGR check
python 02_baseline.py --backbone resnet50     # step 2 of the progression
python 02_baseline.py --backbone dinov2_vits14
python 03_confidence.py                       # when should the estimate be trusted?
```

Plots and metrics land in `out/`. Descriptors are cached in `out/features/`
(gitignored, ~150 MB) keyed on the backbone and the frame list, so re-running an
evaluation is instant; delete the cache after changing the preprocessing.

`vprlib/data.py` is the only place that touches the CSVs — it adds the `session`
and `condition` columns the logger never wrote, and pulls room centres from
`scripts/rooms.py` so the coordinates cannot drift from the rover's.

## Results

Median localisation error / room accuracy, top-1 retrieval:

| Reference → query | ResNet-50 | DINOv2 ViT-S/14 |
|---|---|---|
| **day → night (headline)** | 3.28 m / 0.398 | **0.61 m / 0.668** |
| night → day | 5.28 m / 0.259 | 0.79 m / 0.661 |
| day → held-out day session | 0.27 m / 0.885 | 0.26 m / 0.885 |
| night → night | 0.25 m / 0.882 | 0.28 m / 0.875 |

The two same-condition rows are the ceiling — about 0.26 m, which is roughly the
logger's own 0.2 m frame spacing, so retrieval is finding the right frame nearly
every time. Both backbones hit it; that comparison says nothing about the model.

The headline row is where they separate. ImageNet ResNet-50 features fall apart
across the lighting gap (3.28 m, worse than picking the room at random for
several rooms). DINOv2 keeps the median at 0.61 m — a 5x reduction from the same
pipeline with only the backbone swapped. That gap **is** the result: the
domain-invariance is coming from the self-supervised pretraining, not from
anything in the retrieval code.

Room accuracy of 0.67 lags the metre-level numbers because the naive
nearest-centre labelling assigns corridor frames arbitrarily (42% of all frames
are >2 m from any room centre). It understates real performance and should be
read alongside the metres, not instead of them.

## Knowing when to trust it

For the cross-check job the estimate is only useful with a confidence signal.
Top-1 cosine similarity is a poor one; the spatial spread of the top-5 retrieved
positions is much better — if the five nearest reference frames disagree about
where they are, the match is a coincidence.

Day → night, gating on top-5 spread:

| gate | queries kept | median err | R@1 <1 m | errors >2 m |
|---|---|---|---|---|
| none | 100% | 0.61 m | 0.598 | 30.7% |
| cosine ≥ 0.90 | 13.7% | 0.41 m | 0.766 | 15.8% |
| spread ≤ 0.50 m | 44.3% | 0.32 m | 0.870 | 8.0% |
| spread ≤ 0.25 m | 24.5% | 0.25 m | 0.924 | 5.2% |

Spread dominates: it keeps 3x more queries than the similarity gate while
halving the gross-error rate. At the 0.5 m gate the camera answers 44% of frames
with 0.32 m median error — enough to catch a 4 m Cartographer failure, staying
silent the rest of the time.

`out/error_map_dinov2_vits14.png` shows where the remaining failures live: room
interiors are accurate, and the errors concentrate in the central corridors
between dining, kitchen and entrance — repetitive, low-texture transit space.
That is the complementary failure mode to lidar's, which is the argument in §1.

## Data notes from this pass

- **Channel order is fine.** `out/channel_check.png` compares both
  interpretations; wood and skin read correctly with the file bytes taken as
  RGB. No swap needed — the BGR/RGB worry does not apply to the saved
  JPEGs.
- **`day1_night_session2` has 1,557 clean rows, not 1,556.** Off-by-one in the
  count recorded at collection time. Totals are 13,875 (day 9,638, night 4,237).
- All 13,875 CSV rows have an image on disk; 163 images on disk are orphaned and
  ignored, as expected.
- The long gaps documented in §4 are present and benign (max 324 s in
  `day2_evening_500`). 21 `jump=1` rows across all sessions, all sub-metre.
- The `entrance` ambiguity resolves in favour of `rooms.py`: `(-2.94, 7.20)`
  sits directly on every session's trajectory, and the other candidate does not
  correspond to any dwell point in the data.

## Next

Metric learning (step 3) has a clear target now: the corridor frames and the
30.7% ungated gross-error rate. A triplet fine-tune with positives drawn by pose
distance and negatives from the same session but far away is the natural next
step, trained on the day sessions only so the night set stays honest.
