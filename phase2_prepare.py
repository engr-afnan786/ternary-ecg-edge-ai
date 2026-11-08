"""Prepare MIT-BIH N/V beats for a patient-disjoint ECG experiment. Run at project root."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from urllib.request import urlopen

import numpy as np

BASE = "https://physionet.org/files/mitdb/1.0.0/"
RAW = Path("datasets/raw/mitdb-1.0.0")
OUT = Path("datasets/processed/mitdb_nv_v1")
REPORT = Path("results/phase2_dataset.json")
RECORDS = """100 101 103 105 106 108 109 111 112 113 114 115 116 117 118 119
121 122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 219 220
221 222 223 228 230 231 232 233 234""".split()
VAL = set("103 106 113 119 121 209 221".split())
TEST = set("100 105 117 124 200 210 219 228 233".split())
LABEL = {"N": 0, "V": 1}


def partition(record):
    if record in VAL:
        return "validation"
    if record in TEST:
        return "test"
    return "train"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_split():
    check(len(set(RECORDS)) == len(RECORDS) == 44, "Invalid record manifest")
    check(not VAL & TEST and (VAL | TEST) <= set(RECORDS), "Invalid split")
    owners = {}
    for record in RECORDS:
        patient = "201" if record == "202" else record
        split = partition(record)
        check(owners.setdefault(patient, split) == split, "Patient overlap")
    return dict(Counter(owners.values()))


def extract(signal, samples, symbols):
    windows, labels, centers, skipped = [], [], [], Counter()
    for center, symbol in zip(samples, symbols, strict=True):
        if symbol not in LABEL:
            skipped["other_annotation"] += 1
            continue
        left, right = int(center) - 90, int(center) + 90
        if left < 0 or right > len(signal):
            skipped["edge"] += 1
            continue
        windows.append(signal[left:right])
        labels.append(LABEL[symbol])
        centers.append(center)
    x = np.asarray(windows, dtype=np.float32).reshape(-1, 1, 180)
    check(np.isfinite(x).all(), "Nonfinite ECG value")
    return x, np.array(labels, dtype=np.int64), np.array(centers, dtype=np.int64), skipped


def self_test():
    patients = check_split()
    check(sum(patients.values()) == 43, "Patient count mismatch")
    x, y, centers, skipped = extract(np.arange(240), [89, 90, 100, 150, 151], "VNAVV")
    np.testing.assert_array_equal(x[:, 0, [0, 90, 179]], [[0, 90, 179], [60, 150, 239]])
    np.testing.assert_array_equal(y, [0, 1])
    np.testing.assert_array_equal(centers, [90, 150])
    check(skipped == {"edge": 2, "other_annotation": 1}, "Exclusion counts differ")
    print("Window, label and patient-split checks PASS")


def fetch(name, digest=None):
    path = RAW / name
    if path.exists() and (digest is None or hashlib.sha256(path.read_bytes()).hexdigest() == digest):
        return path
    with urlopen(BASE + name, timeout=90) as response:
        data = response.read()
    if digest:
        check(hashlib.sha256(data).hexdigest() == digest, f"Checksum mismatch for {name}")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def prepare():
    import wfdb
    RAW.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.unlink(missing_ok=True)
    sums = {}
    for line in fetch("SHA256SUMS.txt").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        sums[filename.strip().lstrip("*")] = digest
    manifest = {}
    counts = {s: np.zeros(2, dtype=np.int64) for s in ("train", "validation", "test")}
    for rid in RECORDS:
        split = partition(rid)
        print(f"{rid} [{split}]", flush=True)
        for ext in ("atr", "dat", "hea"):
            name = f"{rid}.{ext}"
            fetch(name, sums[name])
        stem = str(RAW / rid)
        rec = wfdb.rdrecord(stem, channel_names=["MLII"], return_res=32)
        check(rec.fs == 360 and rec.sig_name == ["MLII"] and rec.units == ["mV"], f"Unexpected signal in {rid}")
        ann = wfdb.rdann(stem, "atr")
        x, y, samples, skipped = extract(rec.p_signal[:, 0], ann.sample, ann.symbol)
        folder = OUT / split
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{rid}.tmp.npz"
        dest = folder / f"{rid}.npz"
        np.savez(tmp, x=x, y=y, sample=samples)
        with np.load(tmp, allow_pickle=False) as saved:
            for key, expected in (("x", x), ("y", y), ("sample", samples)):
                np.testing.assert_array_equal(saved[key], expected)
        tmp.replace(dest)
        totals = np.bincount(y, minlength=2)
        counts[split] += totals
        manifest[rid] = dict(split=split, file=str(dest), counts=totals.tolist(), skipped=dict(skipped))
    patients = check_split()
    summary = {}
    for split, totals in counts.items():
        check(min(totals) > 0, f"Missing class in {split}")
        summary[split] = dict(patients=patients[split], N=int(totals[0]), V=int(totals[1]))
        print(split, summary[split])
    report = dict(status="PASS", source=BASE, attribution="Moody & Mark, MIT-BIH Arrhythmia Database v1.0.0, PhysioNet, DOI 10.13026/C2F305",
                  license="Open Data Commons Attribution License v1.0", lead="MLII", units="mV",
                  sampling_hz=360, window="center-90 inclusive through center+90 exclusive",
                  normalization="none", labels=LABEL, excluded_records=["102","104","107","217"],
                  versions={p: importlib.metadata.version(p) for p in ("numpy","wfdb")},
                  python=platform.python_version(), splits=summary, records=manifest)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(REPORT)
    print("Phase 2 PASS |", REPORT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        self_test()
        if not args.self_test:
            prepare()
    except Exception as error:
        sys.exit(f"Phase 2 FAIL | {type(error).__name__}: {error}")
