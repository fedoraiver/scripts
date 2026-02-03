from datasets import *
import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, required=True)
    parser.add_argument("--a", type=str, required=True)
    parser.add_argument("--o", type=str, required=True)
    args = parser.parse_args()

    ds_q = load_dataset("json", data_files=args.q)["train"]
    ds_a = load_dataset("json", data_files=args.a)["train"]
    df_q = ds_q.to_pandas()[["question_id", "question"]]
    df_q.index = df_q["question_id"]
    q_dict = df_q["question"].to_dict()

    def process_add_question(sample):
        qid = sample["question_id"]
        sample["question"] = q_dict.get(qid, "")
        return sample

    ds = ds_a.map(process_add_question, num_proc=64)

    # 根据需求修改
    def process(sample, i):
        records = []
        record = {
            "qry": f"<|image_pad|>\nRepresent the given image with the following question. {sample['question']}",
            "qry_image_path": f"VT-VQA/images/abstract_v002_train2015_{str(sample['image_id']).zfill(12)}.jpg",
            "pos_text": sample["multiple_choice_answer"],
            "pos_image_path": "",
        }
        records.append(record)
        return {"records": records}

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
