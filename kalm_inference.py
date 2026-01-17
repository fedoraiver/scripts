from datasets import load_dataset
import inference_HY

ds = load_dataset(
    "/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data",
    split="train",
    streaming=True,
)

for i, sample in enumerate(ds):
    if i > 100:
        break
    messages = [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n"
            + sample["query"].split("Query:", 1)[1].strip(),
        }
    ]
    print(inference_HY.run_chat(messages))
