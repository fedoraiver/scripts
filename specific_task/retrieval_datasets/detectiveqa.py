import argparse
import json
import re
from pathlib import Path


PARAGRAPH_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def parse_novel_file(novel_path: Path) -> dict[int, str]:
    paragraphs: dict[int, str] = {}
    current_idx: int | None = None
    current_lines: list[str] = []

    with open(novel_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            m = PARAGRAPH_RE.match(line)
            if m:
                if current_idx is not None:
                    paragraphs[current_idx] = " ".join(current_lines).strip()
                current_idx = int(m.group(1))
                first_text = m.group(2).strip()
                current_lines = [first_text] if first_text else []
            elif current_idx is not None:
                stripped = line.strip()
                if stripped:
                    current_lines.append(stripped)

    if current_idx is not None:
        paragraphs[current_idx] = " ".join(current_lines).strip()

    return paragraphs


def split_novel_filename(novel_path: Path) -> tuple[str, str]:
    # 文件名格式: {novel_id}-{novel_name}-{author}.txt
    stem_parts = novel_path.stem.split("-")
    novel_id = stem_parts[0]
    title = stem_parts[1] if len(stem_parts) > 1 else novel_path.stem
    return novel_id, title


def write_jsonl(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_split(base_path: Path, lang: str):
    novel_dir = base_path / f"novel_data_{lang}"
    anno_dir = base_path / f"anno_data_{lang}"
    out_dir = base_path / lang

    corpus_rows: list[dict] = []
    corpus_ids: set[str] = set()

    for novel_path in sorted(novel_dir.glob("*.txt")):
        novel_id, title = split_novel_filename(novel_path)
        paragraphs = parse_novel_file(novel_path)
        for paragraph_idx, paragraph_text in sorted(paragraphs.items()):
            chunk_id = f"{novel_id}_{paragraph_idx}"
            corpus_rows.append(
                {
                    "id": chunk_id,
                    "title": title,
                    "text": paragraph_text,
                }
            )
            corpus_ids.add(chunk_id)

    queries_rows: list[dict] = []
    qrels_rows: list[dict] = []
    query_id = 0

    for anno_subdir in ["human_anno", "AIsup_anno"]:
        subdir = anno_dir / anno_subdir
        if not subdir.exists():
            continue

        for anno_file in sorted(subdir.glob("*.json")):
            with open(anno_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            records = data if isinstance(data, list) else [data]
            for record in records:
                novel_id = str(record.get("novel_id"))
                for question in record.get("questions", []):
                    query_text = str(question.get("question", "")).strip()
                    queries_rows.append({"id": query_id, "text": query_text})

                    chunk_ids: list[str] = []
                    for pos in question.get("clue_position", []):
                        if isinstance(pos, int) and pos > 0:
                            chunk_id = f"{novel_id}_{pos}"
                            if chunk_id in corpus_ids and chunk_id not in chunk_ids:
                                chunk_ids.append(chunk_id)

                    if not chunk_ids:
                        answer_pos = question.get("answer_position")
                        if isinstance(answer_pos, int) and answer_pos > 0:
                            answer_chunk_id = f"{novel_id}_{answer_pos}"
                            if answer_chunk_id in corpus_ids:
                                chunk_ids.append(answer_chunk_id)

                    qrels_rows.append(
                        {"chunk_ids": chunk_ids, "extra": 1, "query_id": query_id}
                    )
                    query_id += 1

    write_jsonl(corpus_rows, out_dir / "corpus.jsonl")
    write_jsonl(queries_rows, out_dir / "queries.jsonl")
    write_jsonl(qrels_rows, out_dir / "qrels.jsonl")

    print(f"✅ [{lang}] corpus.jsonl: {len(corpus_rows)}")
    print(f"✅ [{lang}] queries.jsonl: {len(queries_rows)}")
    print(f"✅ [{lang}] qrels.jsonl: {len(qrels_rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "--p",
        type=str,
        required=True,
        help="detective_qa 数据目录",
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    for lang in ["en", "zh"]:
        build_split(base_path, lang)


if __name__ == "__main__":
    main()
    print("✅ done")
