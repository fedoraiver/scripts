from datasets import *
import os
import argparse
import json
from jinja2 import Template

parser = argparse.ArgumentParser()
parser.add_argument(
    "--p",
    type=str,
    required=True,
)
parser.add_argument(
    "--o",
    type=str,
    required=True,
)
parser.add_argument(
    "--s",
    type=str,
    default="train",
)
args = parser.parse_args()


def main():
    with open(args.p + "/config.json", "r") as f:
        config = json.load(f)

    ds = load_dataset(
        "parquet",
        data_files=args.p + "/data/" + args.s + "*.parquet",
        split=args.s,
        cache_dir="./cache_datasets",
    )

    # 处理函数: 每个数据集都不一样,需要按需修改
    def process(sample, i):
        context = {
            "i": i,
            "sample": sample,
            "config": config,
        }

        def render(s):
            return Template(s).render(**context)

        record = {key: render(value) for key, value in config["record"].items()}
        records = []
        records.append(record)
        return {"records": records}

    ds2 = ds.map(
        process,
        with_indices=True,
        num_proc=64,
    )
    all_items = []
    for rec_list in ds2["records"]:
        all_items.extend(rec_list)
    ds3 = Dataset.from_list(all_items)
    os.makedirs(os.path.dirname(args.o), exist_ok=True)
    ds3.to_json(args.o, lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
    print("✅ done")
