from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_npu
from torch.nn.utils.rnn import pad_sequence
from tvm_ffi import device

model_name_or_path = "/mnt/nvme0/tdy/my_models/HY"
device = "npu"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
)
model = model.to(device)
model.eval()


def run_chat_batch(
    batch_messages: list[list[dict]],
    max_new_tokens: int = 2048,
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


batch_messages = [
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n还款还清了，为什么花呗账单显示还要还款",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n当前交易不支持花呗付款怎么回事",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗分期是从什么时候开始",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n昨天晚上借呗借了***，怎么还没到账",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n我能查看用花呗买什么东西吗",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗分期提前还完不可以吗",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗抽天猫券放哪了",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n用蚂蚁花呗缴费产生能量球吗",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么开不了花呗",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗收钱钱码关闭",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n用花呗 蚂蚁森林能产生能量吗",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n开通蚂蚁花呗需要多少分",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n我用蚂蚁花呗确认收货的时间是不一样的，还款期限也是一样的嘛",
        },
    ],
]
print(run_chat_batch(batch_messages))
