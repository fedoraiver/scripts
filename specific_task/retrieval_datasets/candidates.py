import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunk_prefix(chunk_id: str) -> str:
    # peerqa: {paper_id}__p{pidx}
    if "__p" in chunk_id:
        return chunk_id.rsplit("__p", 1)[0]
    # scifact/loogle/detectiveqa: {doc_or_paper}_{chunk_idx}
    if "_" in chunk_id:
        return chunk_id.split("_", 1)[0]
    # qasper: {paper_id}-{section_or_figure}-{idx|file}
    if "-" in chunk_id:
        return chunk_id.split("-", 1)[0]
    return chunk_id


def build_prefix_to_corpus_ids(corpus_path: Path) -> dict[str, list[str]]:
    prefix_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(corpus_path):
        doc_id = str(row["id"])
        prefix_to_ids[chunk_prefix(doc_id)].append(doc_id)
    return dict(prefix_to_ids)


def extract_answer_ids(qrel_row: dict) -> list[str]:
    if "chunk_ids" in qrel_row and isinstance(qrel_row["chunk_ids"], list):
        return [str(x) for x in qrel_row["chunk_ids"]]
    if "chunk_id" in qrel_row:
        chunk_id = qrel_row["chunk_id"]
        if isinstance(chunk_id, list):
            return [str(x) for x in chunk_id]
        return [str(chunk_id)]
    return []


def build_candidates(corpus_path: Path, qrels_path: Path, output_path: Path):
    prefix_to_ids = build_prefix_to_corpus_ids(corpus_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qrel_row in read_jsonl(qrels_path):
            query_id = qrel_row["query_id"]
            answer_ids = extract_answer_ids(qrel_row)

            prefixes: list[str] = []
            for answer_id in answer_ids:
                p = chunk_prefix(answer_id)
                if p not in prefixes:
                    prefixes.append(p)

            candidates: list[str] = []
            for p in prefixes:
                for doc_id in prefix_to_ids.get(p, []):
                    if doc_id not in candidates:
                        candidates.append(doc_id)

            out_row = {"query_id": query_id, "candidates": candidates}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", "--c", type=str, required=True)
    parser.add_argument("--qrels_path", "--q", type=str, required=True)
    parser.add_argument("--output_path", "--o", type=str, required=True)
    args = parser.parse_args()

    build_candidates(
        corpus_path=Path(args.corpus_path),
        qrels_path=Path(args.qrels_path),
        output_path=Path(args.output_path),
    )
    print(f"✅ candidates.jsonl saved to {args.output_path}")


if __name__ == "__main__":
    main()
