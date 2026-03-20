#!/usr/bin/env python3
"""Parse KaLM inference logs and export dataset error-row statistics as JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_RE = re.compile(r"Processing dataset:\s*([^\r\n]+)")
ERROR_RE = re.compile(
    r"line\s+(?P<row>\d+)\s+translate\s+failed:\s*"
    r"(?P<err_type>[A-Za-z_][\w\.]*)\s*:\s*(?P<msg>.*)",
    flags=re.IGNORECASE,
)


def parse_log(log_text: str) -> dict[str, Any]:
    per_dataset_rows: dict[str, set[int]] = defaultdict(set)
    per_dataset_occ: Counter[str] = Counter()
    per_dataset_err_types: dict[str, Counter[str]] = defaultdict(Counter)
    per_dataset_msgs: dict[str, Counter[str]] = defaultdict(Counter)

    current_dataset = "__UNKNOWN__"

    # tqdm/progress output may use '\r'; parse both '\n' and '\r'.
    for raw_line in re.split(r"[\r\n]+", log_text):
        line = raw_line.strip()
        if not line:
            continue

        ds_match = DATASET_RE.search(line)
        if ds_match:
            current_dataset = ds_match.group(1).strip()

        err_match = ERROR_RE.search(line)
        if not err_match:
            continue

        row = int(err_match.group("row"))
        err_type = err_match.group("err_type")
        msg = err_match.group("msg").strip()

        per_dataset_rows[current_dataset].add(row)
        per_dataset_occ[current_dataset] += 1
        per_dataset_err_types[current_dataset][err_type] += 1
        if msg:
            per_dataset_msgs[current_dataset][msg] += 1

    datasets: dict[str, Any] = {}
    total_unique_rows = 0
    total_occurrences = 0

    for ds_name in sorted(per_dataset_rows.keys()):
        unique_rows = sorted(per_dataset_rows[ds_name])
        total_unique_rows += len(unique_rows)
        total_occurrences += per_dataset_occ[ds_name]

        top_messages = [
            {"message": msg, "count": count}
            for msg, count in per_dataset_msgs[ds_name].most_common(5)
        ]

        datasets[ds_name] = {
            "error_row_count": len(unique_rows),
            "error_rows": unique_rows,
            "error_occurrences": per_dataset_occ[ds_name],
            "error_type_counts": dict(per_dataset_err_types[ds_name]),
            "top_messages": top_messages,
        }

    result: dict[str, Any] = {
        "summary": {
            "dataset_count_with_errors": len(datasets),
            "total_error_row_count": total_unique_rows,
            "total_error_occurrences": total_occurrences,
        },
        "datasets": datasets,
    }
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract dataset error rows from kalm_vllm_inference logs"
    )
    parser.add_argument(
        "--log",
        "-l",
        type=Path,
        required=True,
        help="Path to inference log file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON path (default: <log_stem>_error_stats.json)",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    log_path: Path = args.log
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_path: Path
    if args.output is None:
        output_path = log_path.with_name(f"{log_path.stem}_error_stats.json")
    else:
        output_path = args.output

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    report = parse_log(log_text)
    report["log_path"] = str(log_path)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote error stats JSON: {output_path}")
    print(
        "Datasets with errors:",
        report["summary"]["dataset_count_with_errors"],
        "| Total unique error rows:",
        report["summary"]["total_error_row_count"],
        "| Total error occurrences:",
        report["summary"]["total_error_occurrences"],
    )


if __name__ == "__main__":
    main()
