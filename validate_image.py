from datasets import load_dataset, Dataset
import argparse
import re
from pathlib import Path
from PIL import Image, ImageFile
from utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = False

# 根据需求修改
def process(sample, i):
    img_path = Path(args.image_dir) / sample["qry_image_path"]

    try:
        with Image.open(str(img_path)) as img:
            img.verify()
        with Image.open(str(img_path)) as img:
            img.load()
    except Exception:
        print(f"❌ Broken or truncated image at index {i}: {img_path}")
        try:
            img_path.unlink(missing_ok=True)
        except:
            pass
    return {"records": None}


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--i", type=str, required=True)
    parser.add_argument("--image_dir", "--i_d", type=str, required=True)
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.input, cache_dir="/mnt/nvme0/tdy/cache")[
        "train"
    ]
    ds2 = ds.map(process, with_indices=True, num_proc=DEFAULT_NUM_PROCS)


if __name__ == "__main__":
    main()
    print("✅ done")
