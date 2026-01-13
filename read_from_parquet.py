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
        "--name",
        "--n",
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
        "parquet",
        data_files=str(Path(args.project) / "data" / f"{args.name}*.parquet"),
        split=args.split,
    )

    pprint(ds)
    pprint("---------------------------------------------")
    pprint(ds[0])


if __name__ == "__main__":
    main()
    print("✅ done")
