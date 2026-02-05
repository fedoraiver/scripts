import argparse
import json
from pathlib import Path
from typing import Any


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def jsonl_to_tsv(
    input_path: Path,
    output_path: Path,
    include_header: bool,
    key_order: list[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers: list[str] | None = key_order
    wrote_header = False

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")

            if headers is None:
                headers = list(obj.keys())

            if include_header and not wrote_header:
                fout.write("\t".join(headers) + "\n")
                wrote_header = True

            values = [stringify(obj.get(key, "")) for key in headers]
            fout.write("\t".join(values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert each JSONL line's values into one TSV row."
    )
    parser.add_argument("--input", "--i", type=Path, help="Input .jsonl path")
    parser.add_argument(
        "--output", "--o", type=Path, nargs="?", help="Output .tsv path"
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Write header using keys from selected order",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Key order list, e.g. --keys query_id chunk_ids extra",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_suffix(".tsv")

    jsonl_to_tsv(
        input_path,
        output_path,
        include_header=args.header,
        key_order=args.keys,
    )
    print(f"✅ Converted: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
    print("✅ done")
