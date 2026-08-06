"""ROGII ensemble blend submission (code competition, internet OFF, CPU).

Weights: {'A': 0.85, 'B1': 0.15}
Reads source predictions from the mounted dataset taichiiiii/rogii-ens-sources and writes a
weighted blend to submission.csv aligned to the competition sample_submission
id order. Runs unmodified on Kaggle (/kaggle/input) and locally (validation).
"""
import os
import glob
import numpy as np
import pandas as pd

WEIGHTS = {'A': 0.85, 'B1': 0.15}     # source_key -> weight (only nonzero used)
FILES = {'A': 'sub_A.csv', 'B1': 'sub_B1.csv', 'B2': 'sub_B2.csv'}         # source_key -> filename in the dataset
DATASET_MOUNT = "rogii-ens-sources"
LOCAL_DS = os.environ.get("ROGII_WORK", "work") + "/ens_sources_dataset"
SAMPLE_LOCAL = os.environ.get("ROGII_REPO", ".") + "/data/raw/sample_submission.csv"


def _dbg():
    try:
        dirs = os.listdir("/kaggle/input")
        print("INPUT DIRS:", dirs)
        for d in dirs:
            try:
                print("  ", d, "->", os.listdir("/kaggle/input/" + d)[:8])
            except Exception:
                pass
    except Exception as e:
        print("dbg-skip:", e)


def read_source(key):
    fname = FILES[key]
    cands = ["/kaggle/input/%s/%s" % (DATASET_MOUNT, fname), os.path.join(LOCAL_DS, fname)]
    cands += glob.glob("/kaggle/input/*/%s" % fname)
    cands += glob.glob("/kaggle/input/**/%s" % fname, recursive=True)
    for p in cands:
        if os.path.exists(p):
            df = pd.read_csv(p)[["id", "tvt"]].copy()
            assert df["tvt"].isna().sum() == 0, "source %s has NaN tvt" % key
            assert df["id"].duplicated().sum() == 0, "source %s has dup id" % key
            return df.set_index("id")["tvt"]
    raise FileNotFoundError("source %s (%s) not found; tried %s" % (key, fname, cands))


def output_order(fallback_ids):
    # prefer competition sample_submission id order if discoverable; else source order
    cands = ["/kaggle/input/rogii-wellbore-geology-prediction/sample_submission.csv"]
    cands += glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True) + [SAMPLE_LOCAL]
    for p in cands:
        if os.path.exists(p):
            return pd.read_csv(p)["id"].tolist()
    return fallback_ids


def main():
    _dbg()
    total = None
    for key, w in WEIGHTS.items():
        if w == 0:
            continue
        s = read_source(key)
        total = w * s if total is None else total.add(w * s)  # index-aligned
    assert total is not None and total.isna().sum() == 0, "NaN after blend"
    order = output_order(list(total.index))
    total = total.reindex(order)
    assert total.isna().sum() == 0, "missing ids after reindex to submission order"
    out = pd.DataFrame({"id": total.index, "tvt": total.to_numpy()})
    assert np.isfinite(out["tvt"].to_numpy()).all()
    out.to_csv("submission.csv", index=False)
    print("wrote submission.csv rows=%d mean=%.4f min=%.2f max=%.2f"
          % (len(out), out["tvt"].mean(), out["tvt"].min(), out["tvt"].max()))


if __name__ == "__main__":
    main()
