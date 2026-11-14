"""Create the project dataset (binary classification, breast-cancer features).

    python -m training.make_dataset --version v1        # original 569 rows
    python -m training.make_dataset --version v2        # v1 + 300 noisy augmented rows
    python -m training.make_dataset --version corrupt   # FAILURE INJECTION: nulls + missing label column
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

OUT = Path("data/raw/dataset.csv")


def build(version: str, seed: int = 42) -> pd.DataFrame:
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.rename(columns=lambda c: c.replace(" ", "_"))
    if version == "v1":
        return df
    rng = np.random.default_rng(seed)
    extra = df.sample(300, replace=True, random_state=seed).copy()
    feats = [c for c in df.columns if c != "target"]
    extra[feats] = extra[feats] * rng.normal(1.0, 0.05, size=extra[feats].shape)
    df2 = pd.concat([df, extra], ignore_index=True)
    if version == "v2":
        return df2
    if version == "corrupt":
        bad = df2.drop(columns=["target"]).copy()
        bad.iloc[::7, 3] = np.nan
        return bad
    raise SystemExit(f"unknown version {version}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=["v1", "v2", "corrupt"])
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = build(args.version)
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(frame)} rows, version={args.version})")
