import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "--i",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        "--o",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as json_file:
        # 需要实际根据 JSON 结构调整此行
        data = json.load(json_file)["questions"]
    with open(args.output, "w", encoding="utf-8") as jsonl_file:
        # 需要实际根据 JSON 结构调整此行
        for item in data:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
    print("✅ Done")
