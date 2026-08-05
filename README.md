# ROGII — Wellbore Geology Prediction (final submission)

Final submission code for the Kaggle competition
[ROGII — Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
(Featured, $50,000, RMSE on `tvt`). Team **Latte**.

This branch holds only what produced the two submissions we selected for scoring.
The research that led there — 140+ logged experiments, most of them dead ends — is on
the [`develop`](../../tree/develop) branch.

| selected submission | public LB | change from the base configuration |
|---|---|---|
| `rogii-v2-projdeg2-probe` | **6.394** | trend-fit polynomial degree 3 → 2 |
| `rogii-v2-seeds192-probe` | **6.389** | particle-filter seed count 128 → 192 |

## The task

Each horizontal well carries a 1-D gamma-ray (GR) log along measured depth. A vertical
*type well* gives the GR signature against stratigraphic depth (TVT). Because the
subsurface is layered, the horizontal GR is a locally shifted and stretched copy of the
type-well GR, and predicting TVT means locating, at every lateral point, where the drill
bit sits in the stratigraphic column. It is a signal-registration problem with a
geological prior.

Two measurements shaped everything we did:

- **The error lives in one smooth trend.** `TVT = surface − Z`, and `Z` is known exactly
  inside the evaluation interval, so the high-frequency part is free. The true surface is
  a degree-2 polynomial in measured depth to R² = 0.9925.
- **That trend is not recoverable from GR.** Sub-seismic faulting is not encoded in the
  log. Twenty-two independent neural architectures across several teams all stall around
  RMSE 11, which is also where our four from-scratch learned models stopped.

For scale: a flat prediction scores 15.9, an isolated particle filter or the best neural
network about 11, our submissions 6.39, the competition winner 4.68, and an oracle that
knows the smooth surface exactly 3.9. The gap between 11 and 6.4 is not model capacity —
it is the physically-derived pipeline the public notebook lineage had already built.

## What is in this branch

```
build_final_submissions.py   # rebuilds both selected kernels from the upstream notebook
scripts/submission_status.py # groups submissions by kernel, verifies the final selection
```

`build_final_submissions.py` applies four edits for one kernel and eight for the other,
each anchored on an exact source string, and refuses to run if an anchor is missing or
ambiguous. Verified: the notebooks it emits are byte-identical (SHA-256) to the two that
were actually submitted and scored.

`scripts/submission_status.py --final-check` reads the selection back from the Kaggle API
(`SUBMISSION_GROUP_SELECTED`) and fails unless it matches the plan. Selection can be set
only through the web UI, so an independent read-back is the sole objective proof that the
click landed.

## Provenance, and why the notebook itself is not here

Our submissions are forks of the public Kaggle notebook *ROGGI Physics LB 7.872 v48* by
**evgendvorkin**, itself the product of a long public fork chain (koolbox offline
artefacts, the GS1.30+Q0522 line, and several others). We did not write that pipeline.

That notebook carries no stated open-source license, so this repository does not
redistribute it. Instead the exact patch is published here and the upstream is fetched
from Kaggle at build time — same reproducibility, no redistribution.

```bash
pip install kaggle                      # Kaggle account + API token required
kaggle kernels pull evgendvorkin/rogii-physics-lb-7-872-v48 -p upstream -m
python build_final_submissions.py upstream/rogii-physics-lb-7-872-v48.ipynb -o build
```

Set `owner` in the generated `kernel-metadata.json` to your own Kaggle username before
pushing. Competition data is likewise absent: the ROGII rules permit Competition Use
only, so nothing under `data/` is ever committed.

## What our four edits actually do

Two are portability fixes. Kaggle mounts inputs under two different directory layouts and
which one a worker gets is not predictable; the upstream notebook hardcodes one of them.
On a legacy-layout worker the competition-data constant kills the run outright, while the
ridge-artefact constant sits behind an `.exists()` guard and instead drops a model
component *silently*. Resolving both at run time is a no-op on a current worker and
repairs the other. We lost seven kernel runs to this before finding it.

One relaxes an upstream audit that pins the SHA of a reference submission and raises on
mismatch — any knob that changes predictions trips it. Only the `raise` becomes a
`print`; every structural check in that cell stays.

The last one is the actual experiment: a single constant. `projdeg2` lowers the trend-fit
degree from 3 to 2, which is the one change with independent support from both offline
evaluation and the leaderboard, and it matches the measured degree-2 geometry of the true
surface. `seeds192` raises the particle-filter seed count; that also stops the upstream
seed-branch hedge from firing, so its predictions diverge from the base configuration by
more than the seed count alone — which is precisely why it pairs well with the other.

## Why these two were selected

The final rank uses the better of the two selected submissions, so the pair is worth
`mean − hedge`, where the hedge grows with how differently the two fail. We measured both
terms rather than picking the two best public scores:

- **Mean.** Public draws from this configuration family cluster at 6.389–6.442
  (n = 7, sd 0.017). A separate family we had led with for weeks averaged 6.5195
  (n = 13) — 0.115 worse, giving it a 2.9% chance of being the better of a pair. It was
  dropped.
- **Hedge.** Across all ten candidate pairs, `projdeg2 + seeds192` had both the best mean
  and the highest disagreement between its members. The pair of the two best public
  scores would have been the *worst* choice on the hedge axis, since those two happened
  to be nearly identical predictors.

Differences among the top three pairs are around 0.01, i.e. inside the noise. The one
robust conclusion was to drop the 6.411 draw whose family was 0.115 behind.

## License

Our code is MIT-licensed (see `LICENSE`). It does not extend to the upstream notebook,
which remains under whatever terms its author sets, nor to the competition data.
