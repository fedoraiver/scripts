import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--f",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--o",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    with open(args.f, "r", encoding="utf-8") as json_file:
        # 需要实际根据 JSON 结构调整此行
        data = json.load(json_file)
    with open(args.o, "w", encoding="utf-8") as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
    print("✅ Done")
