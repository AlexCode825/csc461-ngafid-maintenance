#!/usr/bin/env python3
"""
NGAFID pilot: open the benchmark subset ("2days"), inspect it, and train two course-style baselines.

What it does
  1. (optional) downloads and extracts the "2days" benchmark subset (1.1 GB), the same file the
     paper's Zenodo record
  2. prints the structure of the data: columns, label counts, folds, flight lengths, missing values,
     and any column that could identify an aircraft, a date or a maintenance event (needed for RQ2)
  3. builds summary features per flight (mean, std, min, quartiles, max for each of the 23 sensors,
     computed on the last 4096 seconds, like the published benchmark)
  4. runs the provided 5 folds with logistic regression and a random forest (scaling and imputation
     are fitted on the training folds only) and reports accuracy and AUROC per fold
  5. if the file name encodes "flights before maintenance", shows recall by distance to maintenance (RQ1 preview)
  6. writes everything to pilot_report.txt so you can paste it back

Usage (Colab or terminal):
    pip install numpy pandas scikit-learn compress_pickle
    python ngafid_pilot.py --download --data-dir ./ngafid_data
    python ngafid_pilot.py --data-dir ./ngafid_data            # if already downloaded

This is a first look, not the final analysis: the goal is to learn what is really in the data.
"""
import argparse
import os
import pickle
import re
import sys
import urllib.request
import tarfile
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The benchmark subset ("2days", about 1.1 GB) comes from the dataset's Zenodo record,
# doi 10.5281/zenodo.6624956, which is the archive the paper cites. Licence: CC BY 4.0.
# (The authors' own code downloads it from a personal Google Drive link instead; that link
# no longer resolves, so this script uses Zenodo.)
ZENODO_2DAYS = "https://zenodo.org/records/6624956/files/2days.tar.gz?download=1"
# The same record also holds the full dataset, all_flight.tar.gz (4.3 GB), and the dataset
# is mirrored on Kaggle:
#   https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid
# To use a copy you downloaded by hand, skip --download and point --data-dir at the folder
# that holds flight_header.csv and flight_data.pkl.
MAX_LEN = 4096  # the benchmark keeps the last 4096 seconds of each flight
REPORT = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    REPORT.append(msg)


def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    archive = os.path.join(data_dir, "2days.tar.gz")
    if not os.path.exists(archive):
        print("Downloading the '2days' benchmark subset (about 1.1 GB) from Zenodo ...")
        part = archive + ".part"
        try:
            with urllib.request.urlopen(ZENODO_2DAYS) as resp, open(part, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            print()
        except Exception as exc:
            if os.path.exists(part):
                os.remove(part)
            sys.exit(f"Download failed ({exc}). Download 2days.tar.gz by hand from "
                     "https://doi.org/10.5281/zenodo.6624956 and put it in --data-dir.")
        os.rename(part, archive)
    size_mb = os.path.getsize(archive) / 1e6
    if size_mb < 500:
        sys.exit(f"{archive} is only {size_mb:.1f} MB; it should be about 1,100 MB. "
                 "The download was probably an error page. Delete it and try again.")
    print("Extracting ...")
    with tarfile.open(archive) as tar:
        # refuse paths that escape the target directory
        base = os.path.realpath(data_dir)
        for m in tar.getmembers():
            if not os.path.realpath(os.path.join(data_dir, m.name)).startswith(base):
                raise RuntimeError(f"unsafe path in archive: {m.name}")
        tar.extractall(data_dir)


def find_dir(data_dir):
    for dirpath, _, files in os.walk(data_dir):
        if "flight_header.csv" in files and "flight_data.pkl" in files:
            return dirpath
    sys.exit(f"flight_header.csv / flight_data.pkl not found under {data_dir}. Run with --download first.")


def load_pickle(path):
    try:
        from compress_pickle import load
        return load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def get_flight(data, idx):
    try:
        return np.asarray(data[idx], dtype=np.float32)
    except (KeyError, IndexError, TypeError):
        return np.asarray(data[int(idx)], dtype=np.float32)


def summary_features(arr):
    arr = arr[-MAX_LEN:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = np.nanpercentile(arr, [25, 50, 75], axis=0)
        f = [np.nanmean(arr, 0), np.nanstd(arr, 0), np.nanmin(arr, 0), q[0], q[1], q[2], np.nanmax(arr, 0)]
    return np.concatenate(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./ngafid_data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-flights", type=int, default=0, help="debug: use only the first N flights")
    ap.add_argument("--before-value", type=int, default=1,
                    help="value of the before_after column that means 'before maintenance'. It is 1: those rows "
                         "have a negative date_diff and a non-negative number_flights_before. The run prints both, "
                         "so the choice can be checked rather than trusted.")
    args = ap.parse_args()

    if args.download:
        download(args.data_dir)
    d = find_dir(args.data_dir)
    log(f"Data folder: {d}")

    header = pd.read_csv(os.path.join(d, "flight_header.csv"), index_col="Master Index")
    data = load_pickle(os.path.join(d, "flight_data.pkl"))
    if args.max_flights:
        header = header.iloc[: args.max_flights]

    # ------------------------------------------------------------------ structure
    log("\n=== HEADER ===")
    log("shape:", header.shape)
    log("columns:", list(header.columns))
    log(header.head(5).to_string())

    hints = [c for c in header.columns if re.search(r"id|tail|plane|aircraft|date|time|file|event|issue|sn|serial", str(c), re.I)]
    log("\nColumns that might identify aircraft / date / event (needed for grouped splits, RQ2):", hints or "NONE FOUND")

    for col in ["before_after", "hclass", "target_class", "class", "fold"]:
        if col in header.columns:
            log(f"\nvalue counts for '{col}':")
            log(header[col].value_counts(dropna=False).sort_index().to_string())

    # How many flights before the repair this one was. The header carries it directly as
    # 'number_flights_before', with -1 meaning "not applicable" (the flight is after the repair).
    # Older copies of the data only had it inside the file name, hence the fallback.
    if "number_flights_before" in header.columns:
        n_before = pd.to_numeric(header["number_flights_before"], errors="coerce")
        header["n_before"] = n_before.where(n_before >= 0)
        log("\n'number_flights_before' found in the header (-1 = flight is after the repair).")
    elif "filename" in header.columns:
        parts = header["filename"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True).str.split("_")
        header["n_before"] = pd.to_numeric(parts.str[-2], errors="coerce")
        log("\nparsed 'flights before' from the file name (no 'number_flights_before' column).")

    # Which value of before_after means "before the repair"? date_diff settles it: it is the
    # number of days from the repair, so it is negative for flights before it.
    if "before_after" in header.columns and "date_diff" in header.columns:
        log("\nmean date_diff per before_after value (negative = before the repair):")
        log(header.groupby("before_after")["date_diff"].agg(["mean", "min", "max", "count"]).to_string())
    if "before_after" in header.columns and "n_before" in header.columns:
        log("\ncrosstab before_after vs. flights before the repair:")
        log(pd.crosstab(header["before_after"], header["n_before"].clip(upper=10), dropna=False).to_string())

    # ------------------------------------------------------------------ features
    feat_path = os.path.join(args.data_dir, "pilot_features.npz")
    if os.path.exists(feat_path) and not args.max_flights:
        z = np.load(feat_path)
        X, lengths, nan_share, ids = z["X"], z["lengths"], z["nan_share"], z["ids"]
        if not np.array_equal(ids, header.index.values):
            sys.exit("cached features do not match the header: delete pilot_features.npz")
        log("\nloaded cached features")
    else:
        X, lengths, nan_share = [], [], []
        for i, idx in enumerate(header.index):
            arr = get_flight(data, idx)
            lengths.append(len(arr))
            nan_share.append(float(np.isnan(arr).mean()))
            X.append(summary_features(arr))
            if (i + 1) % 1000 == 0:
                print(f"  features {i + 1}/{len(header)}")
        X, lengths, nan_share = np.vstack(X), np.array(lengths), np.array(nan_share)
        if not args.max_flights:
            np.savez(feat_path, X=X, lengths=lengths, nan_share=nan_share, ids=header.index.values)

    log("\n=== SIGNALS ===")
    log(f"flights: {len(header)} | sensors: {X.shape[1] // 7} | feature matrix: {X.shape}")
    log(f"flight length in seconds: min {lengths.min()}, median {int(np.median(lengths))}, max {lengths.max()}")
    log(f"share of missing values per flight: mean {nan_share.mean():.1%}, max {nan_share.max():.1%}")

    # ------------------------------------------------------------------ baselines
    if "before_after" not in header.columns or "fold" not in header.columns:
        log("\nbefore_after or fold column missing: stopping before the baselines.")
    else:
        y = header["before_after"].values
        y_pos = (y == args.before_value).astype(int)  # positive = 'before maintenance'
        log(f"\nUsing positive class = before_after == {args.before_value} ('before maintenance'). "
            "Verify this with the crosstab above.")
        folds = header["fold"].values
        models = {
            "logistic regression (balanced)": lambda: make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced")),
            "random forest": lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0, class_weight="balanced")),
        }
        oof = {name: np.zeros(len(y)) for name in models}
        log("\n=== BASELINES on the provided folds (fit on training folds only) ===")
        for name, make in models.items():
            accs, aucs = [], []
            for f in sorted(np.unique(folds)):
                te = folds == f
                m = make().fit(X[~te], y_pos[~te])
                p = m.predict_proba(X[te])[:, 1]
                oof[name][te] = p
                accs.append(accuracy_score(y_pos[te], p > 0.5))
                aucs.append(roc_auc_score(y_pos[te], p))
            log(f"{name}: accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f} | AUROC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}"
                f"   (per fold AUROC: {', '.join(f'{a:.3f}' for a in aucs)})")
        log("Reference (dataset paper, binary accuracy): InceptionTime 75.5%, Conv-MHSA 76.0%, MiniRocket 59.8%")

        # ------------------------------------------------------------ RQ1 preview
        if "n_before" in header.columns and header["n_before"].notna().any():
            log("\n=== RQ1 PREVIEW: recall (random forest) by number of flights before maintenance ===")
            sel = (y_pos == 1) & header["n_before"].notna().values
            df = pd.DataFrame({"n_before": header["n_before"].values[sel].clip(max=6),
                               "caught": oof["random forest"][sel] > 0.5})
            log(df.groupby("n_before")["caught"].agg(["mean", "count"]).rename(columns={"mean": "recall"}).to_string())
            log("If recall does not rise as n_before gets smaller, the model may not be seeing progressive wear (H1).")

    with open("pilot_report.txt", "w") as f:
        f.write("\n".join(REPORT))
    print("\nSaved pilot_report.txt")


if __name__ == "__main__":
    main()
