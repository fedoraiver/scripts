from hmac import new
from datasets import load_dataset, Dataset
import argparse
from pathlib import Path
import httpx
from openai import OpenAI
import json


MODEL = "HY"
TARGET_LANGUAGE = "Spanish"
CURRENT_DATASET = None

language_table = json.load(
    open(
        "/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data/support_language.json",
        "r",
    )
)
dataset_table = json.load(
    open(
        "/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data/stat.json",
        "r",
    )
)

client1 = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
    http_client=httpx.Client(timeout=httpx.Timeout(3000.0)),
)
client2 = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="EMPTY",
    http_client=httpx.Client(timeout=httpx.Timeout(30000.0)),
)
client3 = OpenAI(
    base_url="http://localhost:8003/v1",
    api_key="EMPTY",
    http_client=httpx.Client(timeout=httpx.Timeout(30000.0)),
)
client4 = OpenAI(
    base_url="http://localhost:8004/v1",
    api_key="EMPTY",
    http_client=httpx.Client(timeout=httpx.Timeout(30000.0)),
)


def contains_chinese(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s[:20])


def translate(content: str, target_language: str, ratio=1.0) -> str:
    system_prompt_other = f"Translate the following segment into {target_language}, without additional explanation."
    system_prompt_zh = f"将以下文本翻译为{language_table[target_language]}，注意只需要输出翻译后的结果，不要额外解释："

    client = None
    length = len(content)

    # if length * ratio < 2730:
    #     client = client1
    # elif length * ratio < 5461:
    #     client = client2
    # else:
    #     client = client3

    client = client1

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    system_prompt_zh
                    if contains_chinese(content)
                    else system_prompt_other
                ),
            },
            {"role": "user", "content": content},
        ],
        extra_body={
            "max_tokens": length * 2,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "temperature": 0.7,
            "top_p": 0.6,
        },
    )

    return resp.choices[0].message.content


def process(sample: dict[str, str | list[str]], idx):
    records = []
    record = sample
    ##根据需求修改########################################################
    record["query"] = (
        record["query"].split("Query:", 1)[0]
        + "Query: "
        + translate(
            record["query"].split("Query:", 1)[1].strip(),
            TARGET_LANGUAGE,
            10000.0 / dataset_table[CURRENT_DATASET]["max_query_len"],
        )
    )
    records.append(record)

    new_pos = []
    for pos_passage in record["pos"]:
        new_pos.append(
            translate(
                pos_passage,
                TARGET_LANGUAGE,
                10000.0 / dataset_table[CURRENT_DATASET]["max_pos_len"],
            )
        )
    record["pos"] = new_pos

    new_neg = []
    for neg_passage in record["neg"]:
        new_neg.append(
            translate(
                neg_passage,
                TARGET_LANGUAGE,
                1000.0 / dataset_table[CURRENT_DATASET]["max_neg_len"],
            )
        )
    record["neg"] = new_neg

    return {"records": records}


def stat_max_len(example):
    max_q = len(example["query"])
    max_p = max(len(x) for x in example["pos"]) if example["pos"] else 0
    max_n = max(len(x) for x in example["neg"]) if example["neg"] else 0
    return {
        "max_q": max_q,
        "max_p": max_p,
        "max_n": max_n,
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path",
        "--p",
        type=str,
        required=True,
    )
    argparser.add_argument("--output", "--o", type=str)
    args = argparser.parse_args()

    kalm_path = Path(args.path)

    results = {}
    for p in kalm_path.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue

        if Path(p / "train.jsonl").exists():
            print("Dataset already processed:", p.name)
            print("-------------------------------------------------------")
            continue

        print(dataset_table[p.name])
        if (
            max(
                dataset_table[p.name]["max_pos_len"],
                dataset_table[p.name]["max_neg_len"],
                dataset_table[p.name]["max_query_len"],
            )
            > 1365
        ):
            print("Skiping dataset due to long passages.")
            print("-------------------------------------------------------")
            continue

        print("Processing dataset:", p.name)
        CURRENT_DATASET = p.name
        ds = load_dataset(path=str(p), split="train")

        # stats = ds.map(
        #     stat_max_len,
        #     num_proc=512,
        # )
        # max_query_len = max(stats["max_q"])
        # max_pos_len = max(stats["max_p"])
        # max_neg_len = max(stats["max_n"])
        # results[p.name] = {
        #     "max_query_len": max_query_len,
        #     "max_pos_len": max_pos_len,
        #     "max_neg_len": max_neg_len,
        #     "num_samples": len(ds),
        # }
        # print(results[p.name])
        # print("-------------------------------------------------------")
        # with open(file=args.output, mode="w", encoding="utf-8") as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2)

        ds1 = ds.map(process, with_indices=True, num_proc=1024)
        all_items = []
        for rec_list in ds1["records"]:
            all_items.extend(rec_list)
        ds2 = Dataset.from_list(all_items)
        ds2.to_json(str(p / Path("train.jsonl")), lines=True, force_ascii=False)
        print("-------------------------------------------------------")
