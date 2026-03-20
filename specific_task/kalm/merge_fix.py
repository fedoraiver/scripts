#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_BASE_DIR = Path("/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update merged.jsonl rows by copying same-key values from fix.jsonl based on error.json rows."
    )
    parser.add_argument(
        "--error-json", required=True, help="Path to error.json (top-level datasets)."
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help=f"Dataset root (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview only; do not write files."
    )
    return parser.parse_args()


def load_error_json(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(obj, dict)
        or "datasets" not in obj
        or not isinstance(obj["datasets"], dict)
    ):
        raise ValueError(
            'Invalid error.json: expected top-level key "datasets" with dict value.'
        )
    return obj


def extract_error_rows(dataset_info) -> List[int]:
    if isinstance(dataset_info, dict):
        rows = dataset_info.get("error_rows", [])
    elif isinstance(dataset_info, list):
        rows = dataset_info
    else:
        rows = []

    out: List[int] = []
    for v in rows:
        try:
            out.append(int(v))
        except Exception:
            continue
    return sorted(set(out))


def read_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    items.append(obj)
                else:
                    items.append({})
            except Exception:
                items.append({})
    return items


def build_row_to_fix(
    error_rows: List[int], fix_objs: List[dict], merged_line_count: int
) -> Tuple[Dict[int, dict], str]:
    row_to_fix: Dict[int, dict] = {}
    if not error_rows or not fix_objs:
        return row_to_fix, "empty"

    # Case 1: fix.jsonl is exactly per-error-row output.
    if len(fix_objs) == len(error_rows):
        for i, row in enumerate(error_rows):
            if 0 <= row < merged_line_count:
                row_to_fix[row] = fix_objs[i]
        return row_to_fix, "by_error_rows"

    # Case 2: fix.jsonl is full file aligned by row index.
    max_row = max(error_rows)
    if len(fix_objs) > max_row:
        for row in error_rows:
            if 0 <= row < merged_line_count:
                row_to_fix[row] = fix_objs[row]
        return row_to_fix, "by_row_index"

    # Case 3: fallback best-effort sequential mapping.
    n = min(len(fix_objs), len(error_rows))
    for i in range(n):
        row = error_rows[i]
        if 0 <= row < merged_line_count:
            row_to_fix[row] = fix_objs[i]
    return row_to_fix, "best_effort"


def merge_one_dataset(
    dataset_dir: Path, error_rows: List[int], dry_run: bool
) -> Dict[str, int | str]:
    merged_path = dataset_dir / "merged.jsonl"
    fix_path = dataset_dir / "fix5.jsonl"

    if not merged_path.exists():
        return {"status": "skip_no_merged", "updated_rows": 0, "updated_keys": 0}
    if not fix_path.exists():
        return {"status": "skip_no_fix", "updated_rows": 0, "updated_keys": 0}

    # Read physical lines directly. Avoid splitlines(), which can split on
    # U+2028/U+2029 inside JSON string values and corrupt JSONL structure.
    with merged_path.open("r", encoding="utf-8") as f:
        merged_lines_raw = list(f)
    fix_objs = read_jsonl(fix_path)
    row_to_fix, mapping_mode = build_row_to_fix(
        error_rows, fix_objs, len(merged_lines_raw)
    )

    if not row_to_fix:
        return {
            "status": f"skip_no_mapping({mapping_mode})",
            "updated_rows": 0,
            "updated_keys": 0,
        }

    out_lines: List[str] = []
    updated_rows = 0
    updated_keys = 0

    for idx, raw in enumerate(merged_lines_raw):
        if idx not in row_to_fix:
            out_lines.append(raw)
            continue

        try:
            merged_obj = json.loads(raw)
        except Exception:
            out_lines.append(raw)
            continue

        fix_obj = row_to_fix[idx]
        if not isinstance(merged_obj, dict) or not isinstance(fix_obj, dict):
            out_lines.append(raw)
            continue

        changed = 0
        for k in list(merged_obj.keys()):
            if k in fix_obj and merged_obj[k] != fix_obj[k]:
                merged_obj[k] = fix_obj[k]
                changed += 1

        if changed > 0:
            updated_rows += 1
            updated_keys += changed
            out_lines.append(json.dumps(merged_obj, ensure_ascii=False) + "\n")
        else:
            out_lines.append(raw)

    if not dry_run:
        with merged_path.open("w", encoding="utf-8") as f:
            f.writelines(out_lines)

    return {
        "status": f"ok({mapping_mode})",
        "updated_rows": updated_rows,
        "updated_keys": updated_keys,
    }


def main() -> None:
    args = parse_args()
    error_obj = load_error_json(Path(args.error_json))
    base_dir = Path(args.base_dir)

    stats = {
        "target_datasets": len(error_obj["datasets"]),
        "processed": 0,
        "skip_no_merged": 0,
        "skip_no_fix": 0,
        "skip_no_mapping": 0,
        "total_updated_rows": 0,
        "total_updated_keys": 0,
    }

    for dataset_name, dataset_info in error_obj["datasets"].items():
        dataset_dir = base_dir / dataset_name
        error_rows = extract_error_rows(dataset_info)
        res = merge_one_dataset(dataset_dir, error_rows, args.dry_run)

        status = str(res["status"])
        if status.startswith("ok("):
            stats["processed"] += 1
            stats["total_updated_rows"] += int(res["updated_rows"])
            stats["total_updated_keys"] += int(res["updated_keys"])
            print(
                f"[OK] {dataset_name}: {status}, updated_rows={res['updated_rows']}, updated_keys={res['updated_keys']}"
            )
        elif status == "skip_no_merged":
            stats["skip_no_merged"] += 1
            print(f"[SKIP] {dataset_name}: {status}")
        elif status == "skip_no_fix":
            stats["skip_no_fix"] += 1
            print(f"[SKIP] {dataset_name}: {status}")
        else:
            stats["skip_no_mapping"] += 1
            print(f"[SKIP] {dataset_name}: {status}")

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
    print("✅ done")
