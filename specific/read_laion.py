import pandas as pd
import requests
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


data_root_dir = "laion400m/data"
images_dir = "./laion400m/images"

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


def download_image(url, save_path):
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return f"✅ 下载完成: {save_path}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ 下载失败: {save_path} -> {e}"


parser = argparse.ArgumentParser()
parser.add_argument("--c", type=int, required=True)
args = parser.parse_args()


def main():
    for m in matches[args.c : args.c + 1]:
        df = pd.read_parquet(m, engine="pyarrow")
        print(f"File: {m}")

        futures = []
        with ThreadPoolExecutor(max_workers=256) as executor:
            for i in df.index:
                url = df.at[i, "url"]
                dir_index = i // 1000
                offset_index = i % 1000
                save_path = f"{images_dir}/{m}/{dir_index}/{offset_index}.jpg"
                futures.append(executor.submit(download_image, url, save_path))

            for future in as_completed(futures):
                print(future.result())


if __name__ == "__main__":
    main()
    print("✅ done")
