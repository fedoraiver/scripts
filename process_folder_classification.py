import os
import json


def generate_jsonl(root_dir, out_file):

    items = []

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)

            if not os.path.isfile(fpath):
                continue

            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                continue

            # 根据实际情况修改
            items.append(
                {
                    "qry_image_path": fpath.replace("/data", "", 1),
                    "pos_text": class_name,
                    "pos_image_path": "",
                    "qry": "<|image_pad|>\nRepresent the given image for classification.",
                }
            )

    with open(out_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    root = "RESISC45/data"
    output = "RESISC45/filtered_data.jsonl"

    generate_jsonl(root, output)
    print("✅ done")
