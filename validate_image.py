from datasets import load_dataset, Dataset
import argparse
import os
import re
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

parser = argparse.ArgumentParser()
parser.add_argument("--input", "--i", type=str, required=True)
parser.add_argument("--image_dir", "--i_d", type=str, required=True)
args = parser.parse_args()


# 根据需求修改
def process(sample, i):
    img_path = os.path.join(
        args.image_dir,
        sample["qry_image_path"],
    )

    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            img.load()
    except Exception:
        print(f"❌ Broken or truncated image at index {i}: {img_path}")
        try:
            os.remove(img_path)
        except:
            pass
    return {"records": None}


def main():
    ds = load_dataset("json", data_files=args.input, cache_dir="/mnt/nvme0/tdy/cache")[
        "train"
    ]
    ds2 = ds.map(process, with_indices=True, num_proc=128)


if __name__ == "__main__":
    main()
    print("✅ done")
