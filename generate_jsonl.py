from datasets import load_dataset
from pathlib import Path
import argparse
import json
from jinja2 import Template
from utils import *

def render(s, context):
    return Template(s).render(**context)


# 处理函数: 每个数据集都不一样,需要按需修改
def process(sample, i):
    records = []
    record = {
        # "query_id": i,
        # "chunk_id": sample["chunk_id"],
        # "extra": 1,
        # "id": i,
        # "text": sample["query"],
        "id": sample["chunk_id"],
        "text": sample["chunk"],
    }
    records.append(record)
    return {"records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "--p",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out",
        "--o",
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

    # with open(args.path + "/config.json", "r") as f:
    #     config = json.load(f)

    ds = load_dataset(
        path=str(Path(args.path)),
        split=args.split,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    # ds = load_dataset(
    #     "parquet",
    #     data_files=args.path + "/data/" + args.split + "*.parquet",
    #     split=args.split,
    # )

    ds2 = ds.map(
        process,
        with_indices=True,
        num_proc=DEFAULT_NUM_PROCS,
    )
    save_mapped_records_jsonl(ds2, args.out)


if __name__ == "__main__":
    main()
    print("✅ done")
