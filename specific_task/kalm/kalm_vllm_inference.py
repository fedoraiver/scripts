from datasets import load_dataset
import random
import argparse
from pathlib import Path
import httpx
from openai import OpenAI
import json
import re
from typing import List

from utils import *

MODEL = "HY"
TARGET_LANGUAGE = "Spanish"
CURRENT_DATASET = None

client1 = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
    http_client=httpx.Client(timeout=httpx.Timeout(30000.0)),
)
client2 = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="EMPTY",
    http_client=httpx.Client(timeout=httpx.Timeout(30000.0)),
)
clients = [client1, client2]

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


def split_text_into_chunks(text: str, max_len: int = 2666) -> List[str]:
    """
    将任意语言长文本分成多个 chunk，每个 chunk长度 <= max_len，
    尽量保持句子完整和顺序，使用标点正则分句。

    Args:
        text: str, 输入长文本
        max_len: int, chunk 最大长度

    Returns:
        List[str]: 分好的 chunk 列表
    """
    # 正则分句（中文、英文标点）
    # 中文：。！？；   英文：.!?;  支持英文引号
    sentence_endings = r"([。！？；!?\.])"
    sentences = re.split(sentence_endings, text)
    # re.split 会保留分隔符在单独元素，需要合并
    merged_sentences = []
    i = 0
    while i < len(sentences):
        s = sentences[i].strip()
        if i + 1 < len(sentences) and re.match(sentence_endings, sentences[i + 1]):
            s += sentences[i + 1].strip()
            i += 1
        if s:
            merged_sentences.append(s)
        i += 1

    # 2️⃣ 累加句子生成 chunk
    chunks = []
    current_chunk = ""
    for sent in merged_sentences:
        # 处理单句超长：先把已有 chunk 落盘，再把该句按 max_len 硬切分
        if len(sent) > max_len:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for i in range(0, len(sent), max_len):
                piece = sent[i : i + max_len].strip()
                if piece:
                    chunks.append(piece)
            continue

        if len(current_chunk) + len(sent) + 1 > max_len:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent
        else:
            current_chunk = current_chunk + " " + sent if current_chunk else sent

    # 3️⃣ 添加最后一个 chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def contains_chinese(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s[:20])


def translate(content: str, target_language: str, idx, ratio=1.0) -> str:
    system_prompt_other = f"Translate the following segment into {target_language}, without additional explanation."
    system_prompt_zh = f"将以下文本翻译为{language_table[target_language]}，注意只需要输出翻译后的结果，不要额外解释："
    client = random.choice(clients)
    length = len(content)

    try:
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
    except Exception as e:
        # Avoid non-picklable exceptions breaking multiprocessing
        print(f"line {idx} translate failed: {type(e).__name__}: {e}")
        return content

    return resp.choices[0].message.content


def process(sample: dict[str, str | list[str]], idx):
    records = []
    record = sample

    prefix, query_text = record["query"].split("Query:", 1)
    query_text = query_text.strip()
    chunks = split_text_into_chunks(query_text)
    translated_chunks = [translate(chunk, TARGET_LANGUAGE, idx) for chunk in chunks]
    record["query"] = prefix + "Query: " + " ".join(translated_chunks)
    records.append(record)

    new_pos = []
    for pos_passage in record["pos"]:
        chunks = split_text_into_chunks(pos_passage)
        translated_chunks = [translate(chunk, TARGET_LANGUAGE, idx) for chunk in chunks]
        new_pos.append(" ".join(translated_chunks))
    record["pos"] = new_pos

    new_neg = []
    for neg_passage in record["neg"]:
        chunks = split_text_into_chunks(neg_passage)
        translated_chunks = [translate(chunk, TARGET_LANGUAGE, idx) for chunk in chunks]
        new_neg.append(" ".join(translated_chunks))
    record["neg"] = new_neg

    return {"records": records}


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path",
        "--p",
        type=str,
        required=True,
    )
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

        print("Processing dataset:", p.name)
        CURRENT_DATASET = p.name
        ds = load_dataset(
            path="parquet",
            data_files=str(p) + "/*.parquet",
            split="train",
            cache_dir="/mnt/nvme0/tdy/cache_datasets",
        )

        ds1 = ds.map(
            process,
            with_indices=True,
            num_proc=DEFAULT_NUM_PROCS - 100,
        )
        with open(str(p / Path("train.jsonl")), "w", encoding="utf-8") as f:
            for rec_list in ds1["records"]:
                for rec in rec_list:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
    print("✅ done")
