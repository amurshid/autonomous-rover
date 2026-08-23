# VPR — visual relocalisation for the Wave Rover

Given a camera frame from inside the house, return a map-frame pose — position
and yaw — by embedding the frame and retrieving the nearest reference frame,
whose pose is known from Cartographer at recording time.

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
python 10_finetune.py --blocks 4 # fine-tune the backbone  (this one worked too)
python 11_finetune_eval.py       # the runs compared on uncontaminated splits
```

The numbered scripts are the study, run once each in order -- `06` diagnoses
`04` and `05`, and `09` tests the prediction `06` makes, so the sequence is part
of the result. Deployment tooling is not part of that sequence and lives in
`deploy/`:

```
python deploy/export.py          # build the Pi bundle (TorchScript + database)
python deploy/replay.py          # replay a session through the deployment code
```

`03` and `08` each evaluate two configurations: the original frozen top-1
measurement, and the shipped one (fine-tuned backbone + the post-processing in
§7). `08` also runs both restricted to the two sessions the fine-tuned model
never trained on, which is the only fair comparison between them.

`10_finetune.py` is the only script here that needs a GPU to be practical — it
cannot cache descriptors, because the backbone changes every step. It
auto-selects `cuda`/`mps`/`cpu`; on an 8-thread CPU expect ~9 min/epoch. On the
M5 Pro's MPS backend a 12-epoch run takes 25–35 minutes end to end, so the
whole `--blocks` sweep is about 1.5 hours.

The environment is in `environment.yml` (`conda env create -f environment.yml`).
`vpr_data/` and `out/features/` are gitignored; the descriptor cache rebuilds
itself in ~3.5 minutes.

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

The shipped configuration is the fine-tuned backbone (§6) plus aggregation and
the sequence filter (§7). It trained on six of the eight sessions, so its
all-session numbers are optimistic; the honest comparison restricts both models
to `day2_evening_700` and `day1_night_session3`, which training never touched.
Those two are the hardest in the set — the near-sunset lap and a night lap — so
both columns sit below their all-session versions.

**Held-out sessions only, 3,791 frames, same code path, only the model and
post-processing differ:**

| | frozen, top-1 | **shipped** |
|---|---|---|
| position error | 0.32 m median | **0.25 m** |
| mean error | 1.05 m | **0.52 m** |
| within 1 m | 83.1% | **90.1%** |
| yaw error | 6.6° median | **6.0°** |
| frames good enough to seed Cartographer (<1 m, <30°) | 75.8% | **81.9%** |
| gate accepts | 71.9% | **88.4%** |
| usable among accepted | 87.3% | **87.6%** |

Across all eight sessions the shipped configuration reads 0.22 m median and
95.0% within 1 m, and the original frozen measurement read 0.29 m / 87.6% —
but those two are not comparable, for the reason above.

**Cold start**, the actual goal — boot at an arbitrary point, drive, take the
first fix the gate accepts. Simulated on the two held-out sessions, 600 boots
each way:

| | frozen, top-1 | **shipped** |
|---|---|---|
| frames until the gate fires | median 0, p90 3 | median **0**, p90 **1** |
| gate never fired within 25 frames | 0.2% | **0.0%** |
| first accepted fix usable (<1 m, <30°) | 79.5% | **84.5%** |
| first accepted fix within 2 m | 96.3% | **98.7%** |
| with one confirming frame required | 91.1% within 1 m | **94.3%** within 1 m, 98.7% within 2 m |

(Over all eight sessions the frozen model reads 87.9% usable and 97.7% within
2 m — the original headline. The held-out-only column is the honest one.)

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

Day → night, the hard split, frozen backbone (the original measurement):

| gate | queries kept | median err | R@1 <1 m | errors >2 m |
|---|---|---|---|---|
| none | 100% | 0.61 m | 0.598 | 30.7% |
| cosine ≥ 0.90 | 13.7% | 0.41 m | 0.766 | 15.8% |
| spread ≤ 0.50 m | 44.3% | 0.32 m | 0.870 | 8.0% |
| spread ≤ 0.25 m | 24.5% | 0.25 m | 0.924 | 5.2% |

Spread dominates: 3× more queries kept than the similarity gate at half the
gross-error rate. It costs nothing — the top-5 are already retrieved.

The same table for the shipped configuration, on `day1_night_session3` alone
(the only night session the fine-tuned model never saw):

| gate | queries kept | median err | R@1 <1 m | errors >2 m |
|---|---|---|---|---|
| none | 100% | 0.25 m | 0.918 | 3.5% |
| spread ≤ 0.50 m | 85.9% | 0.21 m | **0.974** | **0.4%** |
| spread ≤ 0.25 m | 58.7% | 0.17 m | 0.988 | 0.2% |

Nearly twice the coverage at a twentieth of the gross-error rate. The gate is
doing much less work than it used to, because there is much less to reject.

`out/error_map_dinov2_vits14.png` shows where the failures live: room interiors
are accurate and the errors concentrate in the central corridors between dining,
kitchen and entrance — repetitive, low-texture transit space. That is the
complementary failure mode to lidar's, which is the argument for carrying both.

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

---

## 6. Fine-tuning the backbone — the last open question, and it pays (`10`, `11`)

Everything through `09` trained a linear projection on frozen descriptors, which
can only re-weight the 768 numbers DINOv2 already produced. Fine-tuning can
recover a cue the backbone discarded, which is the reason to expect more from
it. Against it: 9,243 training images against 22M parameters, and `06` showed
590k parameters were already enough to overfit onto daylight cues here.

Same regime as `09` — train on four day sessions plus `day1_night_session2`,
validate on `day1_night_session1`, test on `day1_night_session3`. Last N
transformer blocks unfrozen, layer-wise LR decay 0.75, warmup then cosine, 12
epochs, best epoch chosen on validation.

**The clean cross-illuminant split**: `day1_night_session3` only — never trained
on, never selected on — against a **day-only** database, so the illuminant gap
has to be crossed.

| model | trainable | median | R@1 <1 m | p90 | room acc |
|---|---|---|---|---|---|
| frozen DINOv2 | — | 0.71 m | 0.566 | 7.58 | 0.655 |
| linear head, day+night (`09`) | 0.6M | 0.34 m | 0.840 | 1.52 | 0.865 |
| fine-tuned, last 4 blocks | 7.1M | 0.31 m | 0.860 | 1.42 | 0.873 |
| fine-tuned, last 8 blocks | 14.2M | 0.31 m | 0.874 | 1.20 | 0.886 |
| fine-tuned, last 12 blocks | 21.5M | 0.28 m | 0.884 | 1.12 | 0.897 |
| **+ tuned recipe (below)** | 21.5M | **0.27 m** | **0.893** | **1.07** | **0.902** |

Fine-tuning beats the linear head, by 0.06 m of median and 0.044 of R@1 at the
best setting. That clears the 0.02 m noise threshold, though not
by a wide margin — the honest summary is *a real but modest gain on top of the
much larger one that `09` already delivered*. The p90 is the more convincing
column: 1.52 m down to 1.12 m, so the tail of bad retrievals shrinks more than
the median does.

**Capacity kept paying.** R@1 rises monotonically 4 → 8 → 12 blocks with no
turn-down, and the full backbone was best. That is not what `06` would have
predicted, and the reason is the training regime rather than the parameter
count: with both illuminants in training and a night session for validation,
there is no daylight-specific shortcut for the extra capacity to find. Layer-wise
LR decay also means "12 blocks" is not 22M parameters moving freely — block 0
trains at 0.75^11 ≈ 4% of block 11's rate.

Same ordering on the other two evaluations, so nothing is being traded away:

| | frozen | linear (`09`) | ft-4 | ft-8 | ft-12 |
|---|---|---|---|---|---|
| test session, full database | 0.28 m / 0.893 | 0.26 / 0.926 | 0.25 / 0.926 | 0.24 / 0.930 | **0.24 / 0.937** |
| held-out day `evening_700` | 0.40 m / 0.774 | 0.35 / 0.811 | 0.34 / 0.820 | 0.32 / 0.831 | **0.32 / 0.843** |

The day side improves too — `evening_700` is in no training set and is the
weakest session in the data, so it is the honest day-side check.

### Why `11_finetune_eval.py` exists

`10_finetune.py`'s own cross-illuminant table queries *both* held-out night
sessions, and one of them chose the epoch. With margins this small that is worth
separating, so `11` reloads the three saved backbones and reports the splits
apart. The contaminated version reads about 0.01–0.02 m better and R@1 about
0.006–0.008 higher — small, and mostly not selection at all: the frozen backbone,
which selected nothing, shows the same direction (0.59 m on session1 against
0.71 m on session3), so session1 is simply the easier traversal. The numbers
above are the clean ones regardless.

### Scope — unchanged by this result

This is still the **recorded-illuminant** case. Training saw
`day1_night_session2`, and all three night sessions are one evening under the
same lamps. **The unseen-illuminant number is still 0.61 m** from the day→night
split with no night data in training. Fine-tuning was not tested against an
illuminant it had never seen, because the dataset cannot pose that question once
night is in the training set.

### One more training pass, aimed at the tail (the "tuned recipe")

The remaining failures are concentrated: 4% of queries carry 40% of the total
error. Four changes aimed at that specifically, all in `10_finetune.py`:

- `--batch-pairs 128` instead of 64. The loss mines its hardest negative from
  inside the batch, so the batch *is* the negative pool.
- `--hard-negatives 20`: globally-mined negatives, injected into every batch.
  For each anchor, the most *similar* frames recorded more than 5 m away — the
  corridor aliases that the error tail is made of. Mean distance to a mined
  negative is 8.5 m, so these are genuinely different places that look alike.
  `batch_hard_triplet` grew an `n_pairs` argument so these extra rows can be
  scored as candidate negatives without ever being treated as anchors.
- `--anchors-per-epoch 9000`: every anchor each epoch instead of a third.
- `--blur-p 0.25`: motion blur, calibrated in `vprlib/augment.py`. The blurriest
  5% of frames produce ~30% of gross errors, and a 2–5 px linear kernel
  reproduces that band (a longer kernel makes frames blurrier than any real one).

Result on the clean split: 0.28 → 0.27 m, R@1 0.884 → 0.893. **Validation could
not tell the two apart** — both peaked at 0.913 — so by the metric used for
selection this bought nothing. What it bought was tail: in deployment, gross
errors fell from 1.95% to 1.56%, and on the blurriest 5% of frames mean error
went 1.33 → 1.13 m with gross errors 11.5% → 9.4%, which is the augmentation
doing exactly what it was aimed at.

On effort: this run took an hour and produced less than the post-processing in
§7 produced for free. It is kept because it is better on the
tail and no worse anywhere, not because it was a good use of time.

### What it costs to deploy

Nothing extra at inference. The fine-tuned model is the same ViT-S/14 at the
same 4.6 GFLOPs, and it *removes* the projection head rather than adding one.
The cost is shipping an 88 MB checkpoint to the Pi instead of loading stock
DINOv2 weights, and rebuilding the descriptor table with it.

---

## 7. Post-processing: two free steps between retrieval and the pose (`03`, `08`)

Neither involves training. Both reuse the top-5 that the confidence gate
already needs, so they are free at inference. Both live in `vprlib/retrieval.py`.

**Aggregation.** Copying the top-1 pose cannot beat the database's own spacing —
the logger saved a frame every 0.2 m, so the nearest recorded frame is typically
that far from the true camera position even when retrieval is perfect. A
similarity-weighted mean of the top-5 lands between recorded frames, where the
answer usually is.

**Sequence filter.** Frames arrive ~0.2 m apart, so over an n-frame window the
rover can only have moved ~0.2n m. An estimate that disagrees with recent
history by more than that is not motion, it is a bad match; it gets replaced by
the median of the recent estimates. This is what attacks the tail — perceptual
aliasing cannot be resolved from one frame, but a wrong match jumps while a
right one moves smoothly.

On `day1_night_session3` against a day-only database, fine-tuned backbone:

| | mean | median | p90 | >2 m |
|---|---|---|---|---|
| top-1 | 0.620 | 0.272 | 1.07 | 4.13% |
| + aggregation | 0.549 | 0.241 | 0.92 | 4.13% |
| + sequence filter | **0.472** | 0.245 | 0.83 | **3.48%** |

Mean error down 24% and the median untouched, which is the point: the filter
only overrides estimates that are physically impossible, so good frames pass
through unchanged.

### Three mistakes it went through first

**Averaging positions destroys the median.** The first sequence attempt took the
median of the last N *positions*. On a moving rover that smears 0.2 m of travel
per frame into every estimate: at an 8-frame window the median went 0.24 → 0.67 m
even as the tail improved. Rejection, not averaging.

**Feeding corrections back freezes the estimate.** The second attempt compared
each frame against its own *corrected* history. One substitution freezes the
estimate, the rover drives away from it, every subsequent frame then looks like
an outlier and is frozen too. Median error went to 4.0 m. It compares against
raw estimates only.

**An overridden frame's yaw is still wrong.** Yaw comes from the top-1 retrieved
frame, and on an overridden frame that is exactly the bad match the filter just
rejected — so the pose would carry a corrected position with a stale heading.
Those frames are now marked and refused by the gate. That single change took
usable-among-accepted from 85.9% to 87.6% and the cold-start figure from 82.0%
to 84.5%.

### Tuning on the test set, and what it cost

The window and temperature were first chosen by trying four settings and keeping
whichever scored best on `day1_night_session3` — the test set. Selecting instead
on `day1_night_session1` picked a **3**-frame window rather than 8, and the true
test score was 0.472 m rather than the 0.398 m the test-tuned setting reported.

**Selecting on the test set was worth 0.074 m of imaginary accuracy**, on a
change where the setting had looked like it barely mattered. The tail claim was worse: test-tuned
said gross errors halved (4.13% → 2.07%), honest tuning gives 4.13% → 3.48%.
Nothing trained on the test set at any point; the leak was entirely through
which setting got kept.

The gate threshold was *not* successfully retuned. A sweep on `session1` chose
0.15 m, which on the held-out sessions kept only 29% of frames and failed to
fire on 8% of boots. The original 0.5 m remains the operating point.

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
  training runs, the interpolation sweep, the representation sweep, and now
  three fine-tuning runs. The 0.61 m baseline was the first look and is clean.
  Later small margins are worth less than they appear, and the fine-tuning gain
  over `09` (0.06 m of median) is small enough to belong in that category even
  though it clears the 0.02 m noise threshold. The direction is corroborated by
  R@1, p90 and room accuracy all moving with it, and by all three `--blocks`
  settings landing on the same side.
- **The `--blocks` setting was chosen on the test session.** Validation
  preferred 12 blocks too (R@1 0.913 against 0.889 and 0.905), so the choice is
  defensible, but the 4/8/12 comparison is a sweep read off the test set and the
  spread between its rows should not be quoted as a measured effect.
- **The deployment numbers assume the database covers the current lighting.**
  Booting under an illuminant that was never recorded falls back to the harder
  cross-domain case.
- **The shipped configuration trained on six of the eight sessions.** Any
  leave-one-session-out number that includes those six is a memory test, not a
  relocalisation test. Only the `day2_evening_700` and `day1_night_session3`
  rows are honest for it, and `08` reports them separately for that reason.
- **Nothing has been run on the Pi.** DINOv2 ViT-S/14 at 224 is ~4.6 GFLOPs,
  likely 1–2 s per frame there, and the Pi throttles at 80 °C
  with no heatsink. Cold start needs one fix, so running it once at boot rather
  than continuously suits both the use case and the thermal limit. The database
  is 42 MB in RAM and the search itself is negligible. Untested.

## Deployment

`deploy/export.py` builds `out/bundle/` — a 109 MB directory holding the
TorchScript encoder (architecture and weights in one file, so the Pi needs
neither `torch.hub` nor a network), the float16 descriptor database with poses,
20 golden frames as raw JPEG bytes, and a manifest of every preprocessing
constant.

`scripts/vpr_relocalise.py` is the ROS 2 node. It publishes to `/initialpose`,
which `set_initial_pose.py` already bridges to Cartographer, so integration is
one line of `start_localization.sh`. It runs in **shadow mode by default** —
logging what it would have published, alongside Cartographer's pose at the same
timestamp — and only publishes with `--publish`.

Its retrieval logic is a plain `Relocaliser` class with no ROS dependency, so
`deploy/replay.py` can replay a recorded session through the exact
deployment code with that session masked out of the database:

| `day1_night_session3`, 1,839 frames | |
|---|---|
| position | **0.209 m median**, 95.4% within 1 m |
| yaw | 5.2° median, 95.8% within 30° |
| gate accepted | 90.2%, of which **0.00%** were >2 m wrong |
| first accepted frame | #1, error 0.11 m |

Matching `08`'s 0.21 m / 95.5% for the same session, which is the point: the
deployment path is not silently different from the evaluated one.

**The self-test is the safety net.** `Relocaliser.self_test()` re-embeds 20
frames whose descriptors ship in the bundle and requires cosine ≥ 0.9999. It
catches channel order, resize filter, normalisation and a bad model copy in one
check, and the node refuses to start if it fails. Measured cost of each mistake
it guards against: `torch.hub` with no network is total failure, swapped
channels cost 0.02 m, a NEAREST resize 0.01 m, and a second JPEG generation at
quality 92 nothing at all (cosine 0.99974).

## Next

**The modelling is finished.** Backbone fine-tuning was the last open question,
the post-processing above is measured, and `03` and `08` now report the shipped
configuration rather than frozen descriptors. What remains is engineering:

- Copy `out/bundle/` to the Pi, `pip install torch`, run the self-test.
- **Measure real latency.** 18 ms on this Mac's CPU, so 1–2 s on the Pi is the
  expectation — but it is an expectation, not a measurement, and it is the
  number that decides whether the design survives contact.
- **Run shadow mode for several sessions** before publishing anything, and
  write the promotion criteria down before reading the log. Cartographer is not
  ground truth here — it is what this system exists to cross-check — so
  disagreements need judging, not averaging. For real ground truth, park on the
  marked spot in the work room that `start_localization.sh` already names.
- Then `--publish`, and replace the hardcoded pose in `start_localization.sh`.

Two things that would still move the numbers, in order of value:

1. **Two or three laps with the blinds shut and the house lights on, during the
   day.** A third illuminant, ~20 minutes, weather-independent. It is the only
   way to test generalisation to an illuminant that was never recorded, which no
   result here supports.
2. **Proper sequence matching**, scoring short sequences in descriptor space
   rather than post-filtering positions. The remaining gross errors survived a
   3× larger negative pool, explicit mining of the confusable pairs, and blur
   augmentation — good evidence they are not solvable from a single frame.
