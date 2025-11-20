from datasets import *
import argparse
import os
import re
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False


# 根据需求修改
def process(sample, i):
    img_path = os.path.join(
        "/mnt/nvme0/tdy/midtraining_new/images",
        sample["pos_image_path"],
    )
    # 先检查损坏
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True)
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.f)["train"]
    ds2 = ds.map(process, with_indices=True, num_proc=128)


if __name__ == "__main__":
    main()
    print("✅ done")
