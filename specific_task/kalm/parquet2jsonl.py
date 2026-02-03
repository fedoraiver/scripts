from pathlib import Path
from datasets import load_dataset

if __name__ == "__main__":
    kalm_path = Path("/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data/")
    for p in kalm_path.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue

        output_path = p / "origin.jsonl"
        if output_path.exists():
            print("origin already exists:", p.name)
            continue

        print("Processing meta for dataset:", p.name)
        ds = load_dataset(
            path="parquet",
            data_files=str(p) + "/*.parquet",
            split="train",
            cache_dir="/mnt/nvme0/tdy/cache_datasets",
        )
        ds.to_json(str(output_path), lines=True, force_ascii=False)
        print("Saved origin.jsonl for dataset:", p.name)
