import os
from PIL import Image
from typing import Dict, List, Tuple
import argparse

# 三种节点尺寸
TARGET_SIZES = [
    (672, 672),
    (1000, 1000),
    (1344, 1344),
]


def classify_images_by_resolution(
    folder: str,
) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:
    # 结果结构
    result = {"small": [], "medium1": [], "medium2": [], "large": []}

    # 支持的图片后缀
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)

        # 判断是否是文件 & 图片
        if not os.path.isfile(path):
            continue
        if not any(fname.lower().endswith(ext) for ext in exts):
            continue

        # 读取分辨率
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception as e:
            print(f"⚠️ 读取失败: {path} -> {e}")
            continue

        # 分类
        if w * h < 672**2:
            result["small"].append((path, (w, h)))
        elif w * h < 1000**2:
            result["medium1"].append((path, (w, h)))
        elif w * h < 1036**2:
            result["medium2"].append((path, (w, h)))
        else:
            result["large"].append((path, (w, h)))

    return result


# 示例运行
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    for dir in os.listdir(args.p):
        path = os.path.join(args.p, dir)
        if not os.path.isdir(path):
            continue
        print(f"目录: {dir}")
        summary = classify_images_by_resolution(path)
        for key, items in summary.items():
            print(f"    {key}: {len(items)} images")
    print("✅ done")
