from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_npu


def run_chat(
    messages: list[dict],
    model_name_or_path: str = "/mnt/nvme0/tdy/HY",
    device: str = "npu",
    max_new_tokens: int = 2048,
):
    """
    Run chat-style generation with a causal LM.

    Args:
        model_name_or_path: local path or HF repo
        messages: list of {"role": ..., "content": ...}
        device: "cpu" | "cuda" | "npu"
        max_new_tokens: generation length

    Returns:
        output_text: decoded model output
    """
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
