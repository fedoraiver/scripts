import os
import argparse
import imghdr
import concurrent.futures
from PIL import Image

# 映射真实格式 -> 正确扩展名
EXT_MAP = {
    "jpeg": ".jpg",
    "png": ".png",
    "gif": ".gif",
    "webp": ".webp",
    "bmp": ".bmp",
    "tiff": ".tiff",
}


def detect_real_ext(path):
    """
    判断真实文件类型（读取 magic number、文件头)
    返回：正确扩展名（如 ".png"），如果不是图片则返回 None。
    """
    fmt = imghdr.what(path)  # 快速检测
    if fmt in EXT_MAP:
        return EXT_MAP[fmt]

    # 使用 Pillow 进行更精确检测
    try:
        with Image.open(path) as img:
            fmt = img.format.lower()
            if fmt in EXT_MAP:
                return EXT_MAP[fmt]
    except Exception:
        pass

    return None  # 非图片或损坏图片


def process_file(path):
    """
    处理单个文件：
    - 如果是真图片且扩展名正确 -> 保持
    - 如果是真图片但扩展名错误 -> 修复
    - 如果不是图片 -> 删除
    """
    real_ext = detect_real_ext(path)

    if real_ext is None:
        print(f"[DELETE] Not image → {path}")
        try:
            os.remove(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")
        return

    # 当前扩展名
    base, ext = os.path.splitext(path)
    ext = ext.lower()

    # 扩展名正确
    if ext == real_ext:
        return

    # 修复错误扩展名
    new_path = base + real_ext
    print(f"[FIX] {path} -> {new_path}")

    try:
        os.rename(path, new_path)
    except Exception as e:
        print(f"Failed to rename {path}: {e}")


def get_all_files(root):
    """
    递归获取所有文件路径
    """
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            all_files.append(os.path.join(dirpath, name))
    return all_files


def fix_folder(root, max_workers=16):
    """
    多线程处理整个文件夹
    """
    files = get_all_files(root)
    print(f"Total files: {len(files)}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True)
    args = parser.parse_args()
    folder = args.f
    fix_folder(folder, max_workers=32)  # 可调线程数量
