from datasets import load_dataset
import argparse
from pathlib import Path
import json


def dedup_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "--p",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--split",
        "--s",
        type=str,
        default="train",
    )
    args = parser.parse_args()

    ds = load_dataset(
        path=str(Path(args.path)),
        split=args.split,
        cache_dir="/mnt/nvme0/tdy/cache_datasets",
    )

    out_dir = Path(args.path) / Path(args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) corpus.jsonl
    # 兼容 retrieval_gt 的 chunk id 命名:
    # - 文本段落: {paper_id}-{section_name}-{paragraph_idx}
    # - 图表:     {paper_id}-{file}
    corpus_rows = []
    corpus_seen = set()
    for sample in ds:
        paper_id = str(sample["id"])
        title = sample.get("title", "")

        full_text = sample.get("full_text", {})
        section_names = full_text.get("section_name", [])
        section_paragraphs = full_text.get("paragraphs", [])
        for sec_name, para_list in zip(section_names, section_paragraphs):
            sec_name = str(sec_name)
            for para_idx, para_text in enumerate(para_list):
                chunk_id = f"{paper_id}-{sec_name}-{para_idx}"
                if chunk_id in corpus_seen:
                    continue
                corpus_seen.add(chunk_id)
                corpus_rows.append(
                    {"id": chunk_id, "title": title, "text": str(para_text)}
                )

        figures = sample.get("figures_and_tables", {})
        files = figures.get("file", [])
        captions = figures.get("caption", [])
        for file_name, caption in zip(files, captions):
            chunk_id = f"{paper_id}-{file_name}"
            if chunk_id in corpus_seen:
                continue
            corpus_seen.add(chunk_id)
            corpus_rows.append({"id": chunk_id, "title": title, "text": str(caption)})

    corpus_path = out_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in corpus_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 2) queries.jsonl + 3) qrels.jsonl
    queries_rows = []
    qrels_rows = []
    query_id = 0
    corpus_id_set = {row["id"] for row in corpus_rows}
    for sample in ds:
        questions = sample.get("question", [])
        retrieval_gt = sample.get("retrieval_gt", [])
        pair_count = min(len(questions), len(retrieval_gt))
        for i in range(pair_count):
            question_text = str(questions[i])
            chunk_ids = [str(x) for x in retrieval_gt[i]]
            chunk_ids = dedup_keep_order([x for x in chunk_ids if x in corpus_id_set])

            queries_rows.append({"id": query_id, "text": question_text})
            qrels_rows.append({"chunk_ids": chunk_ids, "extra": 1, "query_id": query_id})
            query_id += 1

    queries_path = out_dir / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for row in queries_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    qrels_path = out_dir / "qrels.jsonl"
    with open(qrels_path, "w", encoding="utf-8") as f:
        for row in qrels_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ corpus.jsonl saved to {corpus_path}")
    print(f"✅ queries.jsonl saved to {queries_path}")
    print(f"✅ qrels.jsonl saved to {qrels_path}")


if __name__ == "__main__":
    main()
    print("✅ done")
