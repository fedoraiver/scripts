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

    # ds = load_dataset(
    #     path="parquet",
    #     data_files=str(Path(args.project)) + "/*.parquet",
    #     split=args.split,
    #     cache_dir="/mnt/nvme0/tdy/cache_datasets",
    # )
    # ds = load_dataset(
    #     path=str(Path(args.project)),
    #     name=args.name,
    #     split=args.split,
    #     cache_dir="/mnt/nvme0/tdy/cache_datasets",
    # )
    ds = load_dataset(
        "json",
        data_files=str(Path(args.project) / Path(args.split + ".jsonl.gz")),
        split="train",
        cache_dir="/mnt/nvme0/tdy/cache_datasets",
    )

    pprint(ds)
    pprint("---------------------------------------------")
    pprint(ds[0])


if __name__ == "__main__":
    main()
    print("✅ done")
