from datasets import load_dataset, Dataset
from pathlib import Path
import argparse
import json
from split import *
from typing import Dict, Union


def export_qrels_jsonl(qrels: Dict[str, List[str]], output_path: Union[str, Path]):
    """
    将 qrels 导出为 jsonl，每行 {"chunk_id": str, "extra": 1, "query_id": int}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for qid_str, chunk_ids in qrels.items():
            query_id = int(qid_str)
            for chunk_id in chunk_ids:
                line = {
                    "chunk_id": chunk_id,
                    "extra": 1,  # 固定为 1
                    "query_id": query_id,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


def export_queries_jsonl(queries: Dict[str, str], output_path: Union[str, Path]):
    """
    将 queries 字典导出为 jsonl，每行 {"id": int, "text": str}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for qid_str, text in queries.items():
            line = {"id": int(qid_str), "text": text}  # 转为 int
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def export_corpus_jsonl(corpus: dict, output_path: Union[str, Path]):
    """
    将 corpus 导出为 jsonl，每行 {"id": "docid_idx", "text": "chunk_text"}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id, chunks in corpus.items():
            for chunk_id, chunk_text in chunks.items():
                line = {"id": f"{doc_id}_{chunk_id}", "text": chunk_text}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    "--p",
    type=str,
    required=True,
)
parser.add_argument(
    "--name",
    "--n",
    type=str,
)
parser.add_argument(
    "--split",
    "--s",
    type=str,
    default="train",
)
args = parser.parse_args()


def main():

    ds = load_dataset(
        path=str(Path(args.path)),
        name=args.name,
        split=args.split,
        cache_dir="/mnt/nvme0/tdy/cache_datasets",
    )
    # ds = load_dataset(
    #     "parquet",
    #     data_files=args.path + "/data/" + args.split + "*.parquet",
    #     split=args.split,
    # )

    # {
    # "context": "The original long input texts",
    # "title": "The title of the given document",  //for arxiv paper, we use "title" to refer the identical ID for specific paper
    # "question": "Question to ask based on the given input",
    # "answer": "Groundtruth answer for the question", // for short dependency cloze, the answer is a list ordered by <mask-0>, <mask-1>, ...
    # "evidence": [ "One or more evidence (complete sentences) for answering the question, which are extracted directly from the original input"
    # ],
    # "metadata": "Metadata for the context",
    # "task": "The task for the question answer",
    # "doc_id": "The document ID",
    # "id": "The task id"
    # }

    corpus = {}
    queries = {}
    qrels = {}
    for i, sample in enumerate(ds):
        if sample["doc_id"] not in corpus:
            chunk_dict = split_text_by_sentence(sample["context"])
            corpus[sample["doc_id"]] = chunk_dict
        evidence_indices = find_evidence_chunks(
            sample["evidence"], corpus[sample["doc_id"]]
        )
        queries[f"{i}"] = sample["question"]
        qrels[f"{i}"] = [f"{sample['doc_id']}_{idx}" for idx in evidence_indices]
    export_corpus_jsonl(
        corpus, output_path=Path(args.path) / Path(args.name) / Path("corpus.jsonl")
    )
    export_queries_jsonl(
        queries, output_path=Path(args.path) / Path(args.name) / Path("queries.jsonl")
    )
    export_qrels_jsonl(
        qrels, output_path=Path(args.path) / Path(args.name) / Path("qrels.jsonl")
    )


if __name__ == "__main__":
    main()
    print("✅ done")
