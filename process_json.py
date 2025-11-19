from datasets import *
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True)
    parser.add_argument("--o", type=str, required=True)
    args = parser.parse_args()

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

    # 根据需求修改
    def process(sample, i):
        records = []
        # if sample.get("pos_text") is None:
        #     print(f"⚠️ pos_text is None at index {i}")
        #     return {"records": records}
        record = {
            "qry": sample["qry"],
            "qry_image_path": sample["qry_image_path"].replace("images/", "", 1),
            "pos_text": sample["pos_text"],
            "pos_image_path": sample["pos_image_path"],
        }
        records.append(record)
        return {"records": records}

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
