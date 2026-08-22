# VPR — visual relocalisation for the Wave Rover

Given a camera frame from inside
the house, return a map-frame pose — position and yaw — by embedding the frame
and retrieving the nearest reference frame, whose pose is known from
Cartographer at recording time.

```
python 01_explore.py             # load, verify, trajectories, channel check
python 02_baseline.py --backbone resnet50
python 02_baseline.py --backbone dinov2_vits14
python 03_confidence.py          # when should the estimate be trusted?
python 04_metric_learning.py     # train a projection  (made the test worse)
python 05_augmented_training.py  # + simulated night   (worse again)
python 06_why_training_hurt.py   # diagnosis: the robustness trade-off
python 07_representation.py      # resolution and pooling sweep  (no effect)
python 08_relocalisation.py      # the deployment question
python 09_mixed_training.py      # night in the training set  (this one worked)
python 10_finetune.py --blocks 4 # fine-tune the backbone  (WRITTEN, NOT YET RUN)
```

`10_finetune.py` is the only script here that needs a GPU to be practical — it
cannot cache descriptors, because the backbone changes every step. It
auto-selects `cuda`/`mps`/`cpu`; on an 8-thread CPU expect ~9 min/epoch. See §8
above for the brief.

Plots and metrics land in `out/`. Descriptors are cached in `out/features/`
(gitignored, ~250 MB) keyed on backbone, frame list and config, so re-running an
evaluation is instant; delete the cache after changing preprocessing.

`vprlib/data.py` is the only place that touches the CSVs — it adds the `session`
and `condition` columns the logger never wrote, and pulls room centres from
`scripts/rooms.py` so the coordinates cannot drift from the rover's.

---

## Headline: it does the job it was built for

`08_relocalisation.py`, leave-one-session-out — each session is the query, the
other seven are the database. This is what the rover has on boot.

| | |
|---|---|
| position error | **0.29 m median**, 87.6% within 1 m |
| yaw error | **5.8° median**, 89.5% within 30° |
| frames good enough to seed Cartographer (<1 m, <30°) | **83.6%** |
| same, among frames the confidence gate accepts | **92.4%** |

**Cold start**, the actual goal — boot at an arbitrary point, drive, take the
first fix the gate accepts, over 2,400 simulated boots:

| | |
|---|---|
| frames until the gate fires | median **0**, p90 **3** |
| gate never fired within 25 frames | 0.1% |
| first accepted fix usable (<1 m, <30°) | **87.9%** |
| first accepted fix within 2 m | **97.7%** |
| with one confirming frame required | **92.2%** within 1 m, 98.9% within 2 m |

That is enough to replace the hand-set RViz pose in `start_localization.sh`.
Cartographer does not need a perfect seed, only one close enough that the scan
matcher locks on rather than searching.

The yaw comes free: the retrieved frame carries one in the CSV.

---

## The research result: crossing the lighting gap

The deployment number above lets night queries match night references. To test
*generalisation* you have to withhold them — train on all five day sessions,
test on all three night sessions, nothing shared.

Median localisation error / room accuracy, top-1:

| Reference → query | ResNet-50 | DINOv2 ViT-S/14 |
|---|---|---|
| **day → night (headline)** | 3.28 m / 0.398 | **0.61 m / 0.668** |
| night → day | 5.28 m / 0.259 | 0.79 m / 0.661 |
| day → held-out day session | 0.27 m / 0.885 | 0.26 m / 0.885 |
| night → night | 0.25 m / 0.882 | 0.28 m / 0.875 |

The two same-condition rows are the ceiling — about 0.26 m, roughly the logger's
0.2 m frame spacing, so retrieval is finding the right frame nearly every time.
Both backbones hit it; that comparison says nothing about the model.

The headline row is where they separate. ImageNet ResNet-50 features fall apart
across the lighting gap. DINOv2 keeps the median at 0.61 m — 5× better from the
same pipeline with only the backbone swapped. The domain invariance comes from
the self-supervised pretraining, not from anything in the retrieval code.

---

## Knowing when to trust it

Cross-checking Cartographer needs a confidence signal, or it raises false
alarms against Cartographer and gets ignored. Top-1 cosine similarity is a poor
one. The **spatial spread of the top-5 retrieved positions** is much better: if
the five nearest reference frames disagree about where they are, the match is a
coincidence.

Day → night, the hard split:

| gate | queries kept | median err | R@1 <1 m | errors >2 m |
|---|---|---|---|---|
| none | 100% | 0.61 m | 0.598 | 30.7% |
| cosine ≥ 0.90 | 13.7% | 0.41 m | 0.766 | 15.8% |
| spread ≤ 0.50 m | 44.3% | 0.32 m | 0.870 | 8.0% |
| spread ≤ 0.25 m | 24.5% | 0.25 m | 0.924 | 5.2% |

Spread dominates: 3× more queries kept than the similarity gate at half the
gross-error rate. It costs nothing — the top-5 are already retrieved.

`out/error_map_dinov2_vits14.png` shows where the failures live: room interiors
are accurate and the errors concentrate in the central corridors between dining,
kitchen and entrance — repetitive, low-texture transit space. That is the
complementary failure mode to lidar's, the argument for carrying both.

---

## Four attempts to beat the frozen baseline, all unsuccessful

The fifth one (§5, below) worked. These four are what located the reason.

Worth reading as a group. Nothing improved the day→night number, and the reasons
are more interesting than a win would have been.

### 1. Metric learning on the descriptors (`04`)

Linear projection trained with a batch-hard triplet loss. Positives are the same
place on a *different lap* (a same-session positive is an adjacent frame and
teaches nothing), negatives are >3 m away. Trained on four day sessions,
validated on `day2_evening_700`.

Validating on `evening_700` is what keeps this honest. Same-condition day
retrieval is already at the 0.26 m ceiling and cannot detect an improvement in
lighting robustness. But `evening_700` was shot near sunset and averages 72 grey
levels — **darker than the night sessions themselves** — so it is a genuine
lighting shift that costs nothing from the test set.

Result: validation R@1 0.769 → **0.808**. Night R@1 0.598 → **0.553**.

### 2. Training against simulated night (`05`)

Diagnosis of the above: the two gaps do not point the same way. Measured
centroid distances in DINOv2 space are 0.118 for `evening_700` and 0.197 for
night, and the colour casts differ (R−B of +0.4 by day, +1.7 at night).
`evening_700` is *less sunlight*; night is *tungsten instead of sunlight*.
Correcting the first does not correct the second.

So `vprlib/augment.py` simulates night on the day frames — calibrated to the
real night statistics, landing at mean 85.8 / R−B +2.4 against the actual
86.2 / +1.7 — and the projection is trained to map the simulated-night version
of a frame onto its daylight version. Lighting invariance supervised directly,
using no night data.

Result: validation **0.814**, the best of any model. Night **0.538**, the worst.

### 3. The diagnosis (`06`)

Interpolating between the frozen weights and the trained weights traces the
trade-off. Both runs give the same shape:

| interpolation | val R@1 | night R@1 |
|---|---|---|
| 0.0 (frozen) | 0.769 | 0.598 |
| 0.4 | 0.803 | **0.612** |
| 0.7 | 0.806 | 0.592 |
| 1.0 (fully trained) | **0.814** | 0.538 |

Night peaks at around 40% and then falls off a cliff; validation climbs all the
way. **Validation selects almost exactly the wrong setting.** Choosing on it
gives α ≈ 0.9–1.0 and lands at 0.538–0.569, both below the frozen baseline.

The mechanism is specialisation. DINOv2's descriptor is broad because it was
trained on 142M images with no notion of this house. Re-weighting it toward
whatever separates places across 7,686 *daylight* frames narrows it onto
daylight cues — window light, sun patches, shadow direction. The validation
session still has sunlight so those cues still work there. Night has none.

The small gain at α ≈ 0.4 is real but is **not claimed as a result**: it was read
off the test curve, and adopting it would be selecting on the test set — the
exact mistake the split discipline exists to prevent.

### 4. Resolution and pooling (`07`)

If fitting to the data is the problem, change the representation instead —
nothing is fitted, so nothing can specialise. Swept input resolution and patch
pooling on the frozen backbone, selected on validation, tested on night once.

| config | val R@1 |
|---|---|
| 224² cls+mean (baseline) | 0.758 |
| 224² cls+gem | 0.756 |
| 322×238 cls+mean (native 4:3) | 0.759 |
| 322×238 cls+gem | 0.756 |
| 448×336 cls+gem | 0.753 |

All five within noise. The validation winner scored **0.595** on night against
the baseline's **0.598** — identical. Quadrupling the input resolution changed
nothing, which rules out descriptor fidelity as the bottleneck and points back
at the domain gap itself.

### What this adds up to

Four attempts, no gain. The common cause is that training only ever saw one
illuminant, so it could not learn to ignore the one variable that mattered —
and the split design is what revealed it. Under a random split every one of
these runs would have reported a win.

That diagnosis makes a prediction, which `09` then tests.

---

## 5. Mixing night into training — this one worked (`09`)

If the failures were caused by one illuminant in training rather than by the
method, then putting night data *in* training should fix it without changing
anything else. Same projection, same triplet loss, same code.

```
train  four day sessions + day1_night_session2     (7,686 day + 1,557 night)
val    day1_night_session1                         held-out night traversal
test   day1_night_session3                         never touched
```

Validating on a held-out *night* session repairs the other half of the problem
in `06`: validation is now the same kind of data as the test, so early stopping
aims at what is being measured. 35.6% of mined positives pair a day frame with a
night frame — those are what supervise lighting invariance, from real data
rather than the synthetic approximation in `05`.

**The sharp test.** Query the two never-trained night sessions against a
**day-only** database, so the illuminant gap must actually be crossed. The only
difference between the models is whether training ever saw a night frame:

| model | median | R@1 <1 m | room acc |
|---|---|---|---|
| frozen DINOv2 | 0.68 m | 0.578 | 0.662 |
| trained on day only | 0.86 m | 0.528 | 0.597 |
| **trained on day + 1 night session** | **0.34 m** | **0.843** | **0.870** |

Half the error, and R@1 from 0.578 to 0.843, from adding ~11% more data. The
day-only model still *degrades* (0.578 → 0.528), reproducing `04` exactly — so
the diagnosis holds and the method was never the problem.

It helps the day side too. On held-out `day2_evening_700`: frozen 0.772,
day-only 0.789, **mixed 0.808**.

With the full database (the other seven sessions) on held-out
`day1_night_session3`:

| model | median | R@1 <1 m | p90 |
|---|---|---|---|
| frozen DINOv2 | 0.28 m | 0.893 | 1.05 |
| trained on day only | 0.26 m | 0.924 | 0.86 |
| trained on day + night | 0.26 m | **0.926** | **0.80** |

Gated at spread ≤ 0.5 m it keeps **86.8%** of frames at 0.22 m median, R@1 0.972,
with **0.4%** gross errors — against 77.7% kept and 92.4% usable for the frozen
model in `08`.

### What this does and does not show

**Does:** generalisation to a new traversal under an illuminant that has been
recorded. That is the deployment case, and it is now substantially better.

**Does not:** generalisation to an illuminant never recorded. All three night
sessions were shot on the same evening under the same lamps, so the model has
seen that illuminant — just not that lap. Change the bulbs, or run at 3am with
one lamp on, and this number does not apply.

**The 0.61 m day→night result above remains the unseen-illuminant number** and is
untouched by this experiment. The two claims are different and only one of them
is supported by each result. Do not quote the 0.34 m as cross-lighting
generalisation.

The structural point from the earlier failures still stands, now with a
demonstration attached: you cannot learn to ignore a variable that does not vary
in training. The fix is a second illuminant in the training set, and it is cheap
— two or three laps with the blinds shut and the house lights on would add a
third, available on demand indoors, weather-independent, and would let the same
code be tested for genuine unseen-illuminant generalisation.

---

## Data notes

- **Channel order is fine.** `out/channel_check.png` compares both
  interpretations; wood and skin read correctly with the file bytes taken as
  RGB. The BGR/RGB worry does not apply to the saved JPEGs.
- **`day1_night_session2` has 1,557 clean rows, not 1,556.** Off-by-one in the
  count recorded at collection time. Totals are 13,875 (day 9,638, night 4,237).
- All 13,875 CSV rows have an image on disk; 163 images on disk are orphaned and
  ignored, as expected.
- The long gaps noted during collection are present and benign (max 324 s in
  `day2_evening_500`). 21 `jump=1` rows across all sessions, all sub-metre.
- The `entrance` ambiguity resolves in favour of `rooms.py`: `(-2.94, 7.20)`
  sits on every session's trajectory; `(-12.15, -5.72)` matches no dwell point.
- Room accuracy understates performance. 42% of frames sit >2 m from any room
  centre, so the naive nearest-centre label is arbitrary for corridor frames.
  Read it alongside the metres, never alone.
- `day2_evening_700` is consistently the weakest session (0.772 leave-one-out
  against ~0.89 for the rest). It is the near-sunset lap, the most distinct
  lighting in the set, and the least covered by the others.

## Caveats on the numbers

- **The night set has been evaluated several times** — baseline, confidence, two
  training runs, the interpolation sweep, the representation sweep. The 0.61 m
  baseline was the first look and is clean. Later small margins are worth less
  than they appear.
- **The deployment numbers assume the database covers the current lighting.**
  Booting under an illuminant that was never recorded falls back to the harder
  cross-domain case.
- **Nothing has been run on the Pi.** DINOv2 ViT-S/14 at 224 is ~4.6 GFLOPs,
  likely 1–2 s per frame there, and the Pi throttles at 80 °C
  with no heatsink. Cold start needs one fix, so running it once at boot rather
  than continuously suits both the use case and the thermal limit. The database
  is 42 MB in RAM and the search itself is negligible. Untested.

## Next

The modelling question is answered; what remains is engineering. Port the
extractor and the descriptor table to the Pi, measure real latency, and wire the
gated fix into `start_localization.sh` in place of the manual RViz pose.
