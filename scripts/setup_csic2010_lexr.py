#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

URLS = {
    "train": "https://www.lexr.ai/downloads/datasets/csic2010/http/csic2010_http_train.zip",
    "test": "https://www.lexr.ai/downloads/datasets/csic2010/http/csic2010_http_test.zip",
}
REPO_ZIP_URL = "https://github.com/msudol/Web-Application-Attack-Datasets/archive/refs/heads/master.zip"


def _download(url: str, retries: int = 3, sleep_s: float = 1.0) -> bytes:
    last_err = None
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as r:
                return r.read()
        except (HTTPError, URLError) as e:
            last_err = e
            time.sleep(sleep_s * (i + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to download {url}")


def _extract_first_csv(zbytes: bytes, out_dir: str, tag: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in {tag} archive")
        name = csv_names[0]
        out_path = os.path.join(out_dir, os.path.basename(name))
        with zf.open(name) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path


def _peek_csv_header(zf: zipfile.ZipFile, name: str) -> list[str]:
    with zf.open(name) as f:
        head = f.read(4096)
    line = head.splitlines()[0].decode("utf-8", errors="ignore")
    return [c.strip().strip('"').strip("'").lower() for c in line.split(",")]


def _extract_csic_csv_from_repo(zbytes: bytes, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        candidates = []
        for info in zf.infolist():
            name = info.filename
            lname = name.lower()
            if not lname.endswith(".csv"):
                continue
            if "csic" in lname:
                candidates.append(info)
        if not candidates:
            # fallback: any CSV under CSVData
            for info in zf.infolist():
                lname = info.filename.lower()
                if lname.endswith(".csv") and "csvdata" in lname:
                    candidates.append(info)
        if not candidates:
            raise RuntimeError("No CSIC-related CSV found in repo archive")
        # prefer files with a label-like column, else pick the largest
        label_keys = {"label", "class", "classification", "target", "attack", "anomaly"}
        ranked = []
        for info in candidates:
            try:
                header = _peek_csv_header(zf, info.filename)
            except Exception:
                header = []
            has_label = any(h in label_keys for h in header)
            ranked.append((has_label, info.file_size, info))
        ranked.sort(reverse=True, key=lambda x: (x[0], x[1]))
        info = ranked[0][2]
        out_path = os.path.join(out_dir, "csic2010.csv")
        with zf.open(info) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path


def _find_label_col(df: pd.DataFrame) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for k in ("label", "class", "classification", "target", "attack", "anomaly"):
        if k in cols_lower:
            return cols_lower[k]
    # heuristic: binary column with small unique set
    for col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) <= 2:
            return col
    return None


def _normalize_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    # Map common string labels to 0/1
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "normal": 0,
        "norm": 0,
        "benign": 0,
        "attack": 1,
        "anomalous": 1,
        "anom": 1,
        "anon": 1,
        "anomaly": 1,
        "malicious": 1,
        "1": 1,
        "0": 0,
    }
    return s.map(mapping).fillna(0).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/csic2010", help="Output data directory")
    args = ap.parse_args()

    out_dir = args.out
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    df = None
    try:
        print("Downloading CSIC 2010 CSVs from lexr.ai ...")
        train_zip = _download(URLS["train"])
        test_zip = _download(URLS["test"])

        print("Extracting CSVs ...")
        train_csv = _extract_first_csv(train_zip, raw_dir, "train")
        test_csv = _extract_first_csv(test_zip, raw_dir, "test")

        print("Combining into csic2010.csv ...")
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        df = pd.concat([df_train, df_test], ignore_index=True)
        source = "lexr.ai"
    except Exception as e:
        print(f"Lexr download failed ({e}); falling back to GitHub repo mirror ...")
        repo_zip = _download(REPO_ZIP_URL)
        repo_csv = _extract_csic_csv_from_repo(repo_zip, out_dir)
        df = pd.read_csv(repo_csv)
        source = "github-mirror"

    # Normalize label column name
    label_col = _find_label_col(df)
    if label_col is None:
        raise RuntimeError(f"Could not find a label column. Columns: {list(df.columns)}")
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df["label"] = _normalize_labels(df["label"])

    out_csv = os.path.join(out_dir, "csic2010.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    if source == "lexr.ai":
        print("Raw CSVs:", train_csv, test_csv)
    else:
        print("Source:", source)


if __name__ == "__main__":
    main()
