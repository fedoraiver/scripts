from datasets import load_dataset
import argparse
from utils import *


def process(sample, i):
    records = []
    record = {}
    ##根据需求修改########################################################
    record = {}
    ####################################################################
    records.append(record)
    return {"records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--i", type=str, required=True)
    parser.add_argument("--output", "--o", type=str, required=True)
    parser.add_argument("--tmp", type=str, required=False)
    parser.add_argument("--tmp2", type=str, required=False)
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.input, cache_dir=DEFAULT_CACHE_DIR)[
        "train"
    ]
    ds2 = ds.map(process, with_indices=True, num_proc=DEFAULT_NUM_PROCS)
    save_mapped_records_jsonl(ds2, args.output)


if __name__ == "__main__":
    main()
    print("✅ done")
