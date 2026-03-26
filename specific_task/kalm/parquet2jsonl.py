import json
from pathlib import Path

from datasets import load_dataset


def parquet_to_jsonl_fallback(parquet_files, output_path):
    import pyarrow.parquet as pq

    with output_path.open("w", encoding="utf-8") as f:
        for parquet_file in parquet_files:
            parquet = pq.ParquetFile(parquet_file)
            # Explicit batch_size avoids a pyarrow path that fails on nested
            # list columns with "chunked array outputs".
            for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
                for row in batch.to_pylist():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    kalm_path = Path(
        "/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data/.ignore"
    )
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
        parquet_files = sorted(p.glob("*.parquet"))
        if not parquet_files:
            print("No parquet files found, skip:", p.name)
            continue

        try:
            ds = load_dataset(
                path="parquet",
                data_files=[str(file) for file in parquet_files],
                split="train",
                cache_dir="/mnt/nvme0/tdy/cache_datasets",
            )
            ds.to_json(str(output_path), lines=True, force_ascii=False)
        except Exception as e:
            print(
                "load_dataset failed for",
                p.name,
                f"({type(e).__name__}: {e})",
                "- falling back to pyarrow parquet reader.",
            )
            parquet_to_jsonl_fallback(parquet_files, output_path)
        print("Saved origin.jsonl for dataset:", p.name)
