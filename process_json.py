from datasets import load_dataset, Dataset
import argparse
import os
import re
from PIL import Image
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--input", "--i", type=str, required=True)
parser.add_argument("--output", "--o", type=str, required=True)
parser.add_argument("--tmp", type=str, required=False)
args = parser.parse_args()


def clean_unwanted_pattern(text):
    unwanted_pattern = re.compile(r"http|www|\.com|images/")
    if not isinstance(text, str):
        return text
    is_match = unwanted_pattern.search(text)
    if is_match == None:
        return text
    else:
        return False


def fix_image_ext(path):
    if not path:
        return path

    base, _ = os.path.splitext(path)
    for ext in [".jpg", ".png", ".jpeg", ".webp"]:
        candidate = base + ext
        if os.path.exists(candidate):
            print(candidate)
            return candidate

    print(f"⚠️ File not found: {path}")
    return path


def crop_image(image_path: str, box: list, save_dir: str = "./tmp") -> str:
    os.makedirs(save_dir, exist_ok=True)
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"can't open the image: {image_path}\nerror: {e}")
    x, y, w, h = box
    crop_box = (x, y, x + w, y + h)
    img_w, img_h = img.size
    if not (0 <= x < img_w and 0 <= y < img_h):
        return None
    right = min(x + w, img_w)
    lower = min(y + h, img_h)
    crop_box = (x, y, right, lower)
    cropped = img.crop(crop_box)
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(save_dir, filename)
    cropped.save(save_path, quality=95)
    return save_path


def process(sample, i):
    records = []
    record = {}
    ##根据需求修改########################################################

    record["qry"] = (
        "<|image_pad|>\nRepresent the given image with the following question. "
        + sample["question_string"]
    )
    record["qry_image_path"] = sample["image_index"] + ".png"
    record["pos_text"] = sample["answer"]
    record["pos_image_path"] = ""

    # record["qry"] = (
    #     "<|image_pad|>\nRepresent the given image with the following question. "
    #     + sample["question"]
    # )
    # record["qry_image_path"] = sample["image_filename"]
    # record["pos_text"] = sample["answer"]
    # record["pos_image_path"] = ""
    ####################################################################
    records.append(record)
    return {"records": records}


def main():
    ds = load_dataset("json", data_files=args.input, cache_dir="/mnt/nvme0/tdy/cache")[
        "train"
    ]
    ds2 = ds.map(process, with_indices=True, num_proc=128)
    all_items = []
    for rec_list in ds2["records"]:
        all_items.extend(rec_list)
    ds3 = Dataset.from_list(all_items)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ds3.to_json(args.output, lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
    print("✅ done")
