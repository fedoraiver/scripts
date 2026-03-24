from datasets import load_dataset
import argparse
from pathlib import Path
import httpx
from openai import OpenAI
import json
import re
from typing import List
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from utils import *

MODEL = "HY"
TARGET_LANGUAGE = "French"
CURRENT_DATASET = None
RETRY_PER_CLIENT = 2
OUTPUT_ROOT = Path("/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data-french")
LANGUAGE_CODE_MAP = {
    "Spanish": "es",
    "French": "fr",
}

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
    将任意语言长文本分成多个 chunk,每个 chunk长度 <= max_len
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


def is_target_language(text: str, target_language: str) -> bool:
    language_code = LANGUAGE_CODE_MAP.get(target_language)
    if language_code is None:
        raise ValueError(
            f"Unsupported target language for detection: {target_language}"
        )
    try:
        return detect(text) == language_code
    except LangDetectException:
        return False


def translate(content: str, target_language: str, idx, ratio=1.0) -> str:
    system_prompt_other = f"Translate the following segment into {target_language}, without additional explanation."
    system_prompt_zh = f"将以下文本翻译为{language_table[target_language]}，注意只需要输出翻译后的结果，不要额外解释："
    # Split load stably by sample index, and fallback to the other endpoint on failure.
    start = idx % len(clients)
    client_order = clients[start:] + clients[:start]
    length = len(content)
    # Prefer a larger budget for long chunks; if server says it's too large,
    # we parse the error and retry with the allowed budget.

    last_exc = None
    for ci, client in enumerate(client_order):
        max_completion_tokens = length * 2
        for attempt in range(1, RETRY_PER_CLIENT + 1):
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
                        "max_tokens": max_completion_tokens,
                        "top_k": 20,
                        "repetition_penalty": 1.05,
                        "temperature": 0.7,
                        "top_p": 0.6,
                    },
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_exc = e
                err_name = type(e).__name__
                if err_name == "BadRequestError":
                    msg = str(e)
                    m = re.search(r"\((\d+)\s*>\s*(\d+)\s*-\s*(\d+)\)", msg)
                    if m:
                        # allowed completion tokens ~= max_ctx - input_tokens
                        allowed = max(64, int(m.group(2)) - int(m.group(3)) - 32)
                        if allowed < max_completion_tokens:
                            max_completion_tokens = allowed
                            # Retry once immediately with smaller max_tokens.
                            continue
                    print(f"line {idx} translate failed: {err_name}: {e}")
                    return content
                if attempt < RETRY_PER_CLIENT:
                    continue
                print(
                    f"line {idx} translate failed on client#{(start + ci) % len(clients) + 1} "
                    f"attempt={attempt}: {err_name}: {e}"
                )
    print(f"line {idx} translate failed: {type(last_exc).__name__}: {last_exc}")
    return content


def translate_text(text: str, idx: int, target_language: str) -> str:
    chunks = split_text_into_chunks(text)
    translated_chunks = [translate(chunk, target_language, idx) for chunk in chunks]
    return " ".join(translated_chunks)


def keep_or_translate(text: str, idx: int, target_language: str) -> str:
    if is_target_language(text, target_language):
        return text
    return translate_text(text, idx, target_language)


def keep_or_translate_labeled_text(text: str, idx: int, target_language: str) -> str:
    instruct_query_match = re.match(
        r"^(Instruct:\s+)(.*?)(\s+)(Query:\s+)(.*)$",
        text,
        flags=re.DOTALL,
    )
    if instruct_query_match:
        _, instruct_text, separator, _, query_text = instruct_query_match.groups()
        translated_instruct = keep_or_translate(
            instruct_text.strip(), idx, target_language
        )
        translated_query = keep_or_translate(query_text.strip(), idx, target_language)
        return f"Instruct: {translated_instruct}{separator}Query: {translated_query}"

    query_match = re.match(r"^(Query:\s+)(.*)$", text, flags=re.DOTALL)
    if query_match:
        _, query_text = query_match.groups()
        translated_query = keep_or_translate(query_text.strip(), idx, target_language)
        return f"Query: {translated_query}"

    instruct_match = re.match(
        r"^(Instruct:\s+)(.*)$", text, flags=re.DOTALL
    )
    if instruct_match:
        _, instruct_text = instruct_match.groups()
        translated_instruct = keep_or_translate(
            instruct_text.strip(), idx, target_language
        )
        return f"Instruct: {translated_instruct}"

    return keep_or_translate(text, idx, target_language)


def process(sample: dict[str, str | list[str]], idx):
    records = []
    record = sample

    record["query"] = keep_or_translate_labeled_text(
        record["query"], idx, TARGET_LANGUAGE
    )
    records.append(record)

    new_pos = []
    for pos_passage in record["pos"]:
        new_pos.append(
            keep_or_translate_labeled_text(pos_passage, idx, TARGET_LANGUAGE)
        )
    record["pos"] = new_pos

    new_neg = []
    for neg_passage in record["neg"]:
        new_neg.append(
            keep_or_translate_labeled_text(neg_passage, idx, TARGET_LANGUAGE)
        )
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

        print(dataset_table[p.name])

        print("Processing dataset:", p.name)
        CURRENT_DATASET = p.name
        ds_origin = load_dataset(
            path="json",
            data_files=str(p / "origin.jsonl"),
            split="train",
            cache_dir="/mnt/nvme0/tdy/cache_datasets",
        )

        ds1 = ds_origin.map(
            process,
            with_indices=True,
            num_proc=DEFAULT_NUM_PROCS - 100,
        )
        output_dir = OUTPUT_ROOT / p.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "translated.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for rec_list in ds1["records"]:
                for rec in rec_list:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("Saved to:", output_path)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
    print("✅ done")
