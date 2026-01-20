from vllm import LLM, SamplingParams


prompts = [
    "Translate the following segment into Spanish, without additional explanation.\n\nHello, my name is Bob.",
    "Translate the following segment into Spanish, without additional explanation.\n\nThe president of the United States is me.",
    "Translate the following segment into Spanish, without additional explanation.\n\nThe capital of France is Paris.",
    "Translate the following segment into Spanish, without additional explanation.\n\nThe future of AI is bright.",
]

sampling_params = SamplingParams(
    top_k=20, top_p=0.6, repetition_penalty=1.05, temperature=0.7
)
llm = LLM(model="/mnt/nvme0/tdy/my_models/HY")

structure_prompts = [
    llm.get_tokenizer().apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=False,
    )
    for prompt in prompts
]
print(prompts)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
