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
    args = parser.parse_args()

    # ds = load_dataset(
    #     path="parquet",
    #     data_files=str(Path(args.project)) + "/*.parquet",
    #     split=args.split,
    #     cache_dir="/mnt/nvme0/tdy/cache_datasets",
    # )
    ds = load_dataset(
        path=str(Path(args.project)),
        split=args.split,
        cache_dir="/mnt/nvme0/tdy/cache_datasets",
    )

    pprint(ds)
    pprint("---------------------------------------------")
    pprint(ds[777])


if __name__ == "__main__":
    main()
    print("✅ done")
