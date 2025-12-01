from datasets import *
import argparse
import os
import re
from PIL import Image
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--f", type=str, required=True)
parser.add_argument("--o", type=str, required=True)
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


# 根据需求修改
def process(sample, i):
    records = []
    # if sample["imageId"] is None:
    #     return {"records": []}
    # if sample["box"] is None:
    #     return {"records": []}
    # if sample["expression"] is None:
    #     return {"records": []}
    # qry_image_path = f"./my_datasets/gqa/images/{sample['imageId']}.jpg"
    # record["qry"] = (
    #     f"<|image_pad|>\nSelect the portion of the image that follows the language expressions. {sample['expression']}"
    # )
    # record["pos_text"] = ""
    # pos_image_path = crop_image(
    #     qry_image_path, sample["box"], "./my_datasets/cops-ref/posimages/"
    # )
    # if pos_image_path is None:
    #     return {"records": []}
    # record["pos_image_path"] = pos_image_path
    # record["qry_image_path"] = qry_image_path
    # records.append(record)
    imageID = sample["key"]
    image_path = f"./my_datasets/visualgenome/images/{imageID}.jpg"
    objects = sample["value"]
    for objectID, box in objects:
        expression_key = imageID + "_" + str(objectID)
        record = {}

    return {"records": records}


def main():
    ds = load_dataset("json", data_files=args.f)["train"]
    ds2 = ds.map(process, with_indices=True, num_proc=64)
    all_items = []
    for rec_list in ds2["records"]:
        all_items.extend(rec_list)
    ds3 = Dataset.from_list(all_items)
    os.makedirs(os.path.dirname(args.o), exist_ok=True)
    ds3.to_json(args.o, lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
    print("✅ done")
