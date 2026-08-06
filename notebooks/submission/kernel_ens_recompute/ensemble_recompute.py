"""ROGII RECOMPUTE ensemble (code-comp valid, runs on hidden test).

Unlike the R38 frozen-CSV blends (dead: they read visible-3-well predictions and
crash on the hidden rerun), this kernel RE-EXECUTES both full pipelines against
whatever test is mounted at scoring, then blends their outputs — so it produces
valid predictions for the hidden wells.

  plateau (pilkwang, ~7.074) : W_PLATEAU
  checkpoint-3 (~8.160)      : 1 - W_PLATEAU   (de-correlated honest anchor)

Mechanism: each pipeline (shipped in dataset taichiiiii/rogii-ens-pipelines) is
run as an isolated subprocess with cwd=/kaggle/working; both write their final
scored submission.csv to /kaggle/working/submission.csv, which we capture between
runs, then blend aligned to the competition sample_submission id order.
"""
import os
import sys
import glob
import subprocess
import numpy as np
import pandas as pd

WORK = "/kaggle/working"
W_PLATEAU = 0.75  # plateau weight; (1-W) = checkpoint-3 honest anchor


def find_one(name):
    hits = glob.glob("/kaggle/input/**/%s" % name, recursive=True)
    if not hits:
        raise FileNotFoundError("not found under /kaggle/input: %s" % name)
    return hits[0]


def run_capture(pipe_name, out_csv):
    sub = os.path.join(WORK, "submission.csv")
    if os.path.exists(sub):
        os.remove(sub)
    pipe = find_one(pipe_name)
    print("=== RUN %s (%s) ===" % (pipe_name, pipe), flush=True)
    r = subprocess.run([sys.executable, pipe], cwd=WORK)
    if r.returncode != 0:
        raise RuntimeError("%s exited with code %d" % (pipe_name, r.returncode))
    if not os.path.exists(sub):
        raise RuntimeError("%s produced no submission.csv" % pipe_name)
    df = pd.read_csv(sub)
    assert list(df.columns)[:2] == ["id", "tvt"], "bad columns from %s" % pipe_name
    assert df["tvt"].isna().sum() == 0, "%s produced NaN tvt" % pipe_name
    assert df["id"].duplicated().sum() == 0, "%s produced dup ids" % pipe_name
    df.to_csv(out_csv, index=False)
    os.remove(sub)
    print("%s -> %s  rows=%d mean=%.3f" % (pipe_name, out_csv, len(df), df["tvt"].mean()), flush=True)
    return df


def sample_ids(fallback):
    for p in glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True):
        try:
            return pd.read_csv(p)["id"].astype(str)
        except Exception:
            continue
    return pd.Series(fallback, dtype=str)


def main():
    A = run_capture("plateau_pipeline.py", os.path.join(WORK, "_plateau.csv"))
    B = run_capture("ckpt_pipeline.py", os.path.join(WORK, "_ckpt.csv"))

    sa = A.assign(id=A["id"].astype(str)).set_index("id")["tvt"]
    sb = B.assign(id=B["id"].astype(str)).set_index("id")["tvt"]

    ids = sample_ids(sorted(set(sa.index) | set(sb.index)))
    pa = ids.map(sa)
    pb = ids.map(sb)
    # only blend where both present; if one is missing an id, fall back to the other
    blend = np.where(
        pa.notna() & pb.notna(),
        W_PLATEAU * pa.to_numpy() + (1.0 - W_PLATEAU) * pb.to_numpy(),
        np.where(pa.notna(), pa.to_numpy(), pb.to_numpy()),
    )
    out = pd.DataFrame({"id": ids.to_numpy(), "tvt": blend})
    assert out["tvt"].isna().sum() == 0, "blend has NaN (an id missing from both pipelines)"
    assert np.isfinite(out["tvt"].to_numpy()).all()
    out.to_csv(os.path.join(WORK, "submission.csv"), index=False)
    print("ENSEMBLE submission.csv rows=%d mean=%.3f min=%.1f max=%.1f (w_plateau=%.2f)"
          % (len(out), out["tvt"].mean(), out["tvt"].min(), out["tvt"].max(), W_PLATEAU), flush=True)


if __name__ == "__main__":
    main()
