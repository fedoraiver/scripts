from datasets import *
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--i_d",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--s",
        type=str,
        default="train",
    )
    args = parser.parse_args()

    def save_image(sample, i):
        # 每个数据集的图片字段名称可能不同，请根据实际情况修改此处
        img = sample["image"]["bytes"]
        # if img.mode != "RGB":
        #     img = img.convert("RGB")
        filename = f"{args.i_d}/img_{i}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # img.save(filename)
        with open(filename, "wb") as f:
            f.write(img)

    ds = load_dataset(
        "parquet",
        data_files=args.p + "/data/" + args.s + "*.parquet",
        split=args.s,
        cache_dir="./cache_datasets",
    )

    ds.map(save_image, with_indices=True, num_proc=64)


if __name__ == "__main__":
    main()
    print("✅ done")
