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
        "--n",
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
        data_files=args.p + "/data/" + args.n + "*.parquet",
        split=args.s,
    )

    pprint(ds)
    pprint("---------------------------------------------")
    pprint(ds[0])


if __name__ == "__main__":
    main()
    print("✅ done")
