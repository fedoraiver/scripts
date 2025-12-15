from datasets import load_dataset, Dataset
import os
import argparse
import json
from jinja2 import Template

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

with open(args.path + "/config.json", "r") as f:
    config = json.load(f)


def render(s, context):
    return Template(s).render(**context)


# 处理函数: 每个数据集都不一样,需要按需修改
def process(sample, i):
    records = []
    for qa_pair in sample["qa"]:
        context = {
            "i": i,
            "sample": qa_pair,
            "config": config,
        }

        record = {
            key: render(value, context) for key, value in config["record"].items()
        }
        records.append(record)
    return {"records": records}


def main():

    ds = load_dataset(
        "parquet",
        data_files=args.path + "/data/" + args.split + "*.parquet",
        split=args.split,
    )

    ds2 = ds.map(
        process,
        with_indices=True,
        num_proc=128,
    )
    all_items = []
    for rec_list in ds2["records"]:
        all_items.extend(rec_list)
    ds3 = Dataset.from_list(all_items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds3.to_json(args.out, lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
    print("✅ done")
