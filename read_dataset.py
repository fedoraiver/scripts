from datasets import load_dataset
import argparse
from pprint import pprint
from pathlib import Path

from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "--p",
        type=Path,
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

    # 按需更改
    ds = load_dataset(
        "json",
        data_files=str(Path(args.path) / Path(args.split + ".jsonl.gz")),
        split=args.split,
        cache_dir=DEFAULT_CACHE_DIR,
    )

    pprint(ds)
    pprint("---------------------------------------------")
    pprint(ds[0])


if __name__ == "__main__":
    main()
    print("✅ done")
