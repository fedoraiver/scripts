import jsonlines
import pandas as pd
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def main():
    data_root_dir = "./my_datasets/laion400m/data"
    pattern = re.compile(
        r"part-\d{5}-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-c000\.snappy\.parquet$"
    )
    matches = []
    for dirpath, dirnames, filenames in os.walk(data_root_dir):
        for filename in filenames:
            if pattern.match(filename):
                full_path = os.path.join(dirpath, filename)
                matches.append(full_path)
    matches.sort()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "--i", type=int, required=True)
    args = parser.parse_args()

    root_dir = (
        "midtraining_new/images/laion400m/part-"
        + str(args.index).zfill(5)
        + "-4227e361-38e7-40d5-8822-c6db46ea077c-c000.snappy.parquet"
    )
    output_file = "my_datasets/laion400m/laion_data_fix" + str(args.index) + ".jsonl"
    df = pd.read_parquet(matches[args.index], engine="pyarrow")

    # 使用 jsonlines 打开写入器
    with jsonlines.open(output_file, mode="w") as writer:
        # 遍历 A 下的子文件夹
        for folder_name in sorted(
            os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else x
        ):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue  # 跳过非文件夹

            # 文件夹编号
            try:
                folder_id = int(folder_name)
            except ValueError:
                continue

            # 遍历该文件夹下的所有 jpg 文件
            for file_name in sorted(os.listdir(folder_path)):
                if not file_name.lower().endswith(".jpg"):
                    continue

                # 获取图片编号
                try:
                    image_id = int(os.path.splitext(file_name)[0])
                except ValueError:
                    continue

                image_path = os.path.join(folder_path, file_name)
                image_index = folder_id * 1000 + image_id

                # 生成记录
                record = {
                    "qry": "<|image_pad|>\nFind a caption for the the given image.",
                    "qry_image_path": image_path.replace("laion400m/", "", 1),
                    "pos_text": df.loc[image_index, "caption"],
                    "pos_image_path": "",
                }

                # 写入 jsonl（每次写一行）
                writer.write(record)


if __name__ == "__main__":
    main()
    print("✅ done")
