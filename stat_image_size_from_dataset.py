from PIL import Image
from datasets import load_dataset

eval_data = load_dataset(
    "/mnt/nvme0/hsz/datasets/MMEB-eval",
    "InfographicsVQA",
    split="test",
)
result = {"small": 0, "medium1": 0, "medium2": 0, "large": 0}
for item in eval_data:
    img_path = "/mnt/nvme0/tdy/datasets/eval_images/" + item["qry_img_path"]
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            # 分类
            if w * h < 672**2:
                result["small"] += 1
            elif w * h < 1000**2:
                result["medium1"] += 1
            elif w * h < 1036**2:
                result["medium2"] += 1
            else:
                result["large"] += 1
    except Exception as e:
        print(f"⚠️ 读取失败: {img_path} -> {e}")

print(result)
