import json

fin = "TAT-DQA/tatdqa_fixed.jsonl"
fout = "TAT-DQA/tatdqa_fixed2.jsonl"


def ensure_str_list(x):
    """Ensure x is a list of strings."""
    if isinstance(x, list):
        return [str(v) for v in x]
    else:
        return [str(x)]


with open(fin, "r") as f_in, open(fout, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            print("JSON parse failed:", e)
            continue

        for q in obj.get("questions", []):
            # unify answer
            q["answer"] = ensure_str_list(q.get("answer", []))

            # unify facts
            q["facts"] = ensure_str_list(q.get("facts", []))

            # unify derivation
            if q.get("derivation") is None:
                q["derivation"] = ""
            else:
                q["derivation"] = str(q["derivation"])

            # ensure req_comparison is bool
            if "req_comparison" in q:
                q["req_comparison"] = bool(q["req_comparison"])

        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
print("✅ done")
