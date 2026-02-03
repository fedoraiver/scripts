from datasets import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    "--p",
    type=str,
    required=True,
)
parser.add_argument(
    "--image_dir",
    "--i_d",
    type=str,
    required=True,
)
parser.add_argument(
    "--split",
    "--s",
    type=str,
    default="train",
)
args = parser.parse_args()


def save_image(sample, i):
    # 每个数据集的图片字段名称可能不同，请根据实际情况修改此处

    img = sample["image"]["bytes"]
    filename = f"{args.image_dir}/img_{i}.jpg"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(img)


def main():

    ds = load_dataset(
        "parquet",
        data_files=args.path + "/data/" + args.split + "*.parquet",
        split=args.split,
    )

    ds.map(save_image, with_indices=True, num_proc=128)


if __name__ == "__main__":
    main()
    print("✅ done")
