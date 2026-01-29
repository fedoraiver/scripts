from datasets import load_dataset
from pathlib import Path
import json


if __name__ == "__main__":
    kalm_path = Path("/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data")
    for p in kalm_path.iterdir():
        if p.name.startswith("."):
            print("Skipping hidden folder:", p.name)
            print("-------------------------------------------------------")
            continue
        if not p.is_dir():
            print("Skipping non-directory file:", p.name)
            print("-------------------------------------------------------")
            continue
        train_jsonl = Path(p / "train.jsonl")
        origin_jsonl = Path(p / "origin.jsonl")
        output_jsonl = Path(p / "merged.jsonl")
        if output_jsonl.exists():
            print("Merged dataset already exists:", p.name)
            print("-------------------------------------------------------")
            continue
        if train_jsonl.exists() and origin_jsonl.exists():
            print("Processing dataset:", p.name)

            ds1 = load_dataset(
                "json", data_files=str(train_jsonl), split="train", streaming=True
            )
            ds2 = load_dataset(
                "json", data_files=str(origin_jsonl), split="train", streaming=True
            )

            with open(str(output_jsonl), "w", encoding="utf-8") as f:
                for e1, e2 in zip(ds1, ds2):
                    merged = dict(e1)
                    for k, v in e2.items():
                        if k in merged:
                            merged["origin_" + k] = v
                        else:
                            merged[k] = v

                    f.write(json.dumps(merged, ensure_ascii=False) + "\n")

            print("Merged dataset saved to:", str(output_jsonl))
            print("-------------------------------------------------------")
        else:
            print("Required files not found in:", p.name)
            print("-------------------------------------------------------")

    print("All datasets processed.")
