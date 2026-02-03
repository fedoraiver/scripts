from datasets import load_dataset
import argparse
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
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

    # 加载数据集
    ds = load_dataset(
        "json",
        data_files=str(Path(args.project) / Path(args.split + ".jsonl.gz")),
        split=args.split,
        cache_dir="/mnt/nvme0/tdy/cache_datasets",
    )

    # 1️⃣ 生成 corpus.jsonl
    corpus_dict = {}  # key=docid, value=text
    for sample in ds:
        for passage in sample.get("positive_passages", []):
            docid = passage["docid"]
            if docid not in corpus_dict:
                corpus_dict[docid] = passage["text"]
        for passage in sample.get("negative_passages", []):
            docid = passage["docid"]
            if docid not in corpus_dict:
                corpus_dict[docid] = passage["text"]

    corpus_path = Path(args.project) / "corpus.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for docid, text in corpus_dict.items():
            json_line = {"id": docid, "text": text}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    # 2️⃣ 生成 queries.jsonl
    queries_path = Path(args.project) / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for sample in ds:
            json_line = {"id": sample["query_id"], "text": sample["query"]}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    # 3️⃣ 生成 qrels.jsonl
    qrels_path = Path(args.project) / "qrels.jsonl"
    with open(qrels_path, "w", encoding="utf-8") as f:
        for sample in ds:
            query_id = sample["query_id"]
            for passage in sample.get("positive_passages", []):
                chunk_id = passage["docid"]
                json_line = {"chunk_id": chunk_id, "extra": 1, "query_id": query_id}
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"✅ corpus.jsonl saved to {corpus_path}")
    print(f"✅ queries.jsonl saved to {queries_path}")
    print(f"✅ qrels.jsonl saved to {qrels_path}")


if __name__ == "__main__":
    main()
    print("✅ done")
