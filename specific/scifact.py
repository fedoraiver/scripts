from datasets import *
import argparse
from pprint import pprint

from pathlib import Path


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
    parser.add_argument(
        "--name",
        "--n",
        type=str,
    )
    args = parser.parse_args()

    ds = load_dataset(
        "json",
        data_files=str(Path(args.project) / Path(args.split + ".jsonl.gz")),
        split="train",
        cache_dir="/mnt/nvme0/tdy/cache_datasets",
    )

    """
    Dataset({
    features: ['query_id', 'query', 'positive_passages', 'negative_passages'],
    num_rows: 809
    })

    positive_passages和negative_passages形如:
    [{"docid":str,"text":str,"title":str},...]
    先根据positive_passages和negative_passages生成corpus.jsonl,形如:
    {"id":str,"text":str},其中id是positive_passages和negative_passages中出现过的的docid
    
    然后根据query_id和query生成queries.jsonl,形如:
    {"id":int,"text":str},其中id是query_id
    
    最后根据query_id和positive_passages生成qrels.jsonl,形如:
    {"chunk_id":str,"extra":1,"query_id":int},其中chunk_id是"{docid}"
    """


if __name__ == "__main__":
    main()
    print("✅ done")
