import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from dipa.io import save_json


COLUMNS = {"method", "participant", "patch_id", "trial", "expected_identity",
           "clean_identity", "clean_detected", "predicted_identity", "detected", "face_confidence"}


def boolean(value):
    if value not in {"0", "1"}:
        raise ValueError("Detection flags must be 0 or 1")
    return value == "1"


def summarize(rows):
    groups, seen = defaultdict(list), set()
    for row in rows:
        if not COLUMNS.issubset(row):
            raise ValueError(f"Missing columns: {COLUMNS - set(row)}")
        row = {key: value.strip() if isinstance(value, str) else value for key, value in row.items()}
        key = tuple(row[k] for k in ("method", "participant", "patch_id", "trial"))
        if not all(key) or key in seen:
            raise ValueError(f"Empty or duplicate trial key: {key}")
        seen.add(key)
        if not row["expected_identity"]:
            raise ValueError("expected_identity cannot be empty")
        clean_detected, detected = boolean(row["clean_detected"]), boolean(row["detected"])
        confidence = float(row["face_confidence"])
        if not math.isfinite(confidence):
            raise ValueError("face_confidence must be finite; use the camera's documented scale")
        if not detected and row["predicted_identity"]:
            raise ValueError("An undetected face must have an empty predicted_identity")
        valid = clean_detected and row["clean_identity"] == row["expected_identity"]
        changed = row["predicted_identity"] != row["clean_identity"]
        groups[row["method"]].append((valid, detected, changed, confidence, row["participant"]))
    if not groups:
        raise ValueError("No trial rows found")
    output = {}
    for method, records in sorted(groups.items()):
        valid = [r for r in records if r[0]]
        n = len(valid)
        detected = sum(r[1] for r in valid)
        successful = sum(r[1] and r[2] for r in valid)
        output[method] = {
            "trials": len(records), "eligible_trials": n,
            "excluded_clean_failures": len(records) - n,
            "participants": len({r[4] for r in valid}),
            "identity_change_rate": sum(r[2] for r in valid) / n if n else None,
            "detected_dodging_rate": successful / n if n else None,
            "dodging_rate_given_detection": successful / detected if detected else None,
            "detection_rate": detected / n if n else None,
            "mean_face_confidence": sum(r[3] for r in valid) / n if n else None,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description="Summarize camera trials, retaining detection failures separately.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file already exists")
    with args.csv.open(newline="") as stream:
        results = summarize(list(csv.DictReader(stream)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
