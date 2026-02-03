from datasets import *
import argparse
import os
import pandas as pd
from PIL import Image
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--e", type=str, required=True)
parser.add_argument("--i", type=str, required=True)
parser.add_argument("--o", type=str, required=True)
parser.add_argument("--tmp", type=str, required=False)
args = parser.parse_args()


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


def process(sample, idx):
    records = []
    record = {}
    if sample["image_id"] is None:
        return {"records": []}
    if sample["box"] is None:
        return {"records": []}
    if sample["expression"] is None:
        return {"records": []}
    qry_image_path = f"./my_datasets/visualgenome/images/{sample['image_id']}.jpg"
    record["qry"] = (
        f"<|image_pad|>\nSelect the portion of the image that follows the language expressions. {sample['expression'][0]}"
    )
    record["pos_text"] = ""
    pos_image_path = crop_image(
        qry_image_path, sample["box"], "./my_datasets/kbref/posimages/"
    )
    if pos_image_path is None:
        return {"records": []}
    record["pos_image_path"] = pos_image_path
    record["qry_image_path"] = qry_image_path
    records.append(record)
    return {"records": records}


def add_key(example):
    return {"key": f"{example['image_id']}_{example['object_id']}"}


def main():
    expression_ds = load_dataset("json", data_files=args.e)["train"]
    expression_ds = expression_ds.map(add_key, num_proc=128)
    image_object_ds = load_dataset("json", data_files=args.i)["train"]
    image_object_ds = image_object_ds.map(add_key, num_proc=128)
    df1 = pd.DataFrame(expression_ds).set_index("key")
    df2 = pd.DataFrame(image_object_ds).set_index("key")
    df_joined = df1.join(df2, how="inner", lsuffix="", rsuffix="_right").reset_index()
    joined = Dataset.from_pandas(df_joined)
    joined.to_json(args.tmp, lines=True, force_ascii=False)
    join_ds = load_dataset("json", data_files=args.tmp)["train"]
    ds = join_ds.map(process, with_indices=True, num_proc=128)
    all_items = []
    for rec_list in ds["records"]:
        all_items.extend(rec_list)
    ds2 = Dataset.from_list(all_items)
    os.makedirs(os.path.dirname(args.o), exist_ok=True)
    ds2.to_json(args.o, lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
    print("✅ done")
