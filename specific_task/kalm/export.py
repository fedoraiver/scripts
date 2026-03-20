from datasets import load_dataset
from pathlib import Path
import math


SHARD_SIZE = 50000


if __name__ == "__main__":
    kalm_path = Path("/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data")
    output_path = Path(
        "/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data-spanish"
    )
    for p in kalm_path.iterdir():
        if p.name.startswith("."):
            print("Skipping hidden folder:", p.name)
            print("-------------------------------------------------------")
            continue
        if not p.is_dir():
            print("Skipping non-directory file:", p.name)
            print("-------------------------------------------------------")
            continue
        merged_jsonl = Path(p / Path("merged.jsonl"))

        if not merged_jsonl.exists():
            print("Skipping, merged.jsonl not found in:", p.name)
            print("-------------------------------------------------------")
            continue
        print("Processing dataset:", p.name)

        out_dataset_path = output_path / p.name
        out_dataset_path.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(
            "json",
            data_files=str(merged_jsonl),
            split="train",
            cache_dir="/mnt/nvme0/tdy/cache_datasets",
        )

        num_samples = len(ds)
        num_shards = math.ceil(num_samples / SHARD_SIZE)

        print("total samples:", num_samples)
        print("num shards:", num_shards)

        for i in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)

            filename = f"train-{i:05d}-of-{num_shards:05d}.parquet"
            shard.to_parquet(out_dataset_path / filename)

            print("saved:", filename)

        print("finished processing dataset:", p.name)
        print("-------------------------------------------------------")

    print("All datasets processed.")
