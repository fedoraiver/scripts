from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="HY",
    messages=[
        {
            "role": "system",
            "content": "Translate the following segment into Spanish, without additional explanation.\n\n",
        },
        {"role": "user", "content": "今天中午吃包子."},
    ],
    extra_body={
        "max_tokens": 32,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_p": 0.6,
    },
)

print(resp.choices[0].message.content)
