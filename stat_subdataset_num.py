from datasets import load_dataset
from pathlib import Path


if __name__ == "__main__":
    path = Path("/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data")
    for p in path.iterdir():
        if p.is_file():
            continue
        if p.name.startswith("."):
            continue
        print(p.name)
        ds = load_dataset(path=str(p), split="train")
        print(len(ds))
        print("---------------")
