from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_npu


def run_chat(
    messages: list[dict],
    model_name_or_path: str = "/mnt/nvme0/tdy/my_models/HY",
    device: str = "npu",
    max_new_tokens: int = 2048,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = model.to(device)
    model.eval()

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
        )

    output_text = tokenizer.decode(outputs[0])

    return output_text


prompt = [
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n当前交易不支持花呗付款怎么回事",
    },
]
print(run_chat(prompt))
