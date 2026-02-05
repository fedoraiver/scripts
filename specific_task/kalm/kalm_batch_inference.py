from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import argparse
import json
from pathlib import Path

batch_size = 32
device = "npu"
model_name_or_path = "/mnt/nvme0/tdy/my_models/HY"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
)
model = model.to(device)
model.eval()


def run_chat_batch(
    batch_messages: list[list[dict]],
    max_new_tokens: int = 4096,
):

    # ===== 1. 每个对话单独 apply_chat_template =====
    input_ids_list = []
    for messages in batch_messages:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )[0]
        input_ids_list.append(ids)

    # ===== 2. padding 到同一长度 =====
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=pad_token_id,
    ).to(device)

    # ===== 3. attention_mask（Ascend 上必须！）=====
    attention_mask = (input_ids != pad_token_id).to(device)

    # ===== 4. batch generate =====
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    # ===== 5. 拆 batch 输出 =====
    results = []
    for i, output in enumerate(outputs):
        prompt_len = input_ids[i].shape[0]
        text = tokenizer.decode(
            output[prompt_len:],
            skip_special_tokens=True,
        )
        results.append(text)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "--p",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    kalm_path = Path(args.path)
    for p in kalm_path.iterdir():
        ds = load_dataset(
            path=str(p),
            split="train",
        )

        with open(p / Path("train.jsonl"), "w") as f_out:
            for batch in ds.iter(batch_size=batch_size):
                messages_batch = []
                for sample in batch["query"]:
                    messages = [
                        {
                            "role": "user",
                            "content": "Translate the following segment into Spanish, without additional explanation.\n\n"
                            + sample.split("Query:", 1)[1].strip(),
                        },
                    ]
                    messages_batch.append(messages)
                qry = run_chat_batch(messages_batch)
                qry = [
                    batch["query"][i].split("Query:", 1)[0] + "Query:" + qry[i]
                    for i in range(len(qry))
                ]

                # messages_batch = []
                # for i, sample in enumerate(batch["pos"]):
                #     messages = [
                #         {
                #             "role": "user",
                #             "content": "Translate the following segment into Spanish, without additional explanation.\n\n"
                #             + sample[j],
                #         }
                #         for j in range(len(sample))
                #     ]
                #     messages_batch.append(messages)
                # pos = run_chat_batch(messages_batch)

                # messages_batch = []
                # for i, sample in enumerate(batch["neg"]):
                #     messages = [
                #         {
                #             "role": "user",
                #             "content": "Translate the following segment into Spanish, without additional explanation.\n\n"
                #             + sample[j],
                #         }
                #         for j in range(len(sample))
                #     ]
                #     messages_batch.append(messages)
                # neg = run_chat_batch(messages_batch)

                # ===== write =====
                for i in range(len(qry)):
                    item = {
                        "query": qry[i],
                        "pos": batch["pos"][i],
                        "neg": batch["neg"][i],
                    }
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
    print("✅ done")
