import json
import csv
from dataclasses import asdict


def write_optimal_split_report(split_stats, output_dir):
    report_dir = output_dir / "composite_scoring"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(stat) for stat in split_stats]

    json_path = report_dir / "optimal_split_calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = report_dir / "optimal_split_calibration.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return [str(json_path), str(csv_path)]
