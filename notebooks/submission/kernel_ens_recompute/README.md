# kernel_ens_recompute — RECOMPUTE ensemble (valid on hidden test)

The R38 frozen-CSV blends (`kernel_ens_*`) are DEAD: they read visible-3-well
predictions from a dataset and crash on the hidden rerun (see ledger R39). This
kernel is the valid replacement — it RE-EXECUTES both full pipelines against
whatever test is mounted at scoring, then blends.

## Mechanism
`ensemble_recompute.py` runs, as isolated subprocesses (cwd=/kaggle/working):
1. `plateau_pipeline.py` (= pilkwang, recompute plateau ~7.074)
2. `ckpt_pipeline.py` (= checkpoint-only, recompute checkpoint-3 ~8.160)
captures each one's `/kaggle/working/submission.csv`, then writes
`0.75*plateau + 0.25*checkpoint3` aligned to the mounted sample_submission ids.

## Pipelines dataset (`taichiiiii/rogii-ens-pipelines`)
Built by extracting the code cells of the two source notebooks
(`kernel_pilkwang_hedge/rogii-pilkwang-hedge.ipynb`,
`kernel_checkpoint_only/rogii-checkpoint-only.ipynb`), stripping `%`/`!` magics,
and prepending `display`/`get_ipython` no-op shims. Both nbconvert-clean, no
`__main__` guard / `sys.argv` / `exit()`. Regenerate + `kaggle datasets version`
if the source kernels change.

## Verified (2026-07-18, visible run)
COMPLETE; log shows both pipelines ran + `ENSEMBLE submission.csv rows=14151
(w_plateau=0.75)`; output 14151 rows, no NaN; blend-vs-pilkwang RMSE 1.315 ≈
0.25*(A-B1 5.32) = correct 0.75/0.25 mix. Submitted ref 54796227.
