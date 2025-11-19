import json

fin = "./TAT-DQA/tatdqa_dataset_train.jsonl"
fout = "./TAT-DQA/tatdqa_fixed.jsonl"

with open(fin, "r") as f_in, open(fout, "w") as f_out:
    for line in f_in:
        obj = json.loads(line)
        for q in obj.get("questions", []):
            if not isinstance(q["answer"], list):
                q["answer"] = [q["answer"]]
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ done")
