from datasets import load_dataset
import json

ds1 = load_dataset("json", data_files="a.jsonl", split="train", streaming=True)
ds2 = load_dataset("json", data_files="b.jsonl", split="train", streaming=True)

with open("out.jsonl", "w", encoding="utf-8") as f:
    for e1, e2 in zip(ds1, ds2):
        merged = dict(e1)
        for k, v in e2.items():
            if k in merged:
                merged["b_" + k] = v
            else:
                merged[k] = v

        f.write(json.dumps(merged, ensure_ascii=False) + "\n")
