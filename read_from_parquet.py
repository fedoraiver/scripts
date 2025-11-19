from datasets import *
import argparse
from pprint import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--s",
        type=str,
        default="train",
    )
    args = parser.parse_args()

    ds = load_dataset(
        "parquet",
        data_files=args.p + "/data/" + args.s + "*.parquet",
        split=args.s,
        cache_dir="./cache_datasets",
    )

    pprint(ds)
    pprint("---------------------------------------------")
    for i in range(5):
        pprint(ds[i])
        pprint("---------------------------------------------")


if __name__ == "__main__":
    main()
    print("✅ done")
