import gc
import torch
from datasets import load_dataset
from more_itertools import distribute
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

ds = load_dataset(
    "/mnt/nvme0/tdy/my_datasets/KaLM-embedding-finetuning-data",
    split="train",
    streaming=True,
)


def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
    model="/mnt/nvme0/tdy/my_models/HY",
    tensor_parallel_size=4,
    distributed_executor_backend="mp",
    max_model_len=4096,
)
prompt_batch = [
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n还款还清了，为什么花呗账单显示还要还款",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n当前交易不支持花呗付款怎么回事",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗分期是从什么时候开始",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n昨天晚上借呗借了***，怎么还没到账",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我能查看用花呗买什么东西吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗分期提前还完不可以吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗抽天猫券放哪了",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n用蚂蚁花呗缴费产生能量球吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么开不了花呗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗收钱钱码关闭",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n用花呗 蚂蚁森林能产生能量吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n开通蚂蚁花呗需要多少分",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我用蚂蚁花呗确认收货的时间是不一样的，还款期限也是一样的嘛",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n如何增加？蚂蚁借呗额度",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗怎么的看借款总金额",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗退款还要收费吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁花呗还钱，，***号之前还款，那九号当天还可以吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n怎么要求花呗退款",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我是还没发退款的，退款到了花呗还是金额没变动",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗怎么利息不同",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n密付可以当使用花呗吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗逾期被停用怎么才能恢复使用",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n借呗利率为什么和别人的不一样",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n怎样设置信用卡花呗收款",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n如果花呗，忘记还款，自己在余额宝里扣除",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n怎么能降低蚂蚁借呗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗 还款怎么还",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗钱己经还了，后面退了款的钱退到那里",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n***号当天花呗消费是算当月的，还是次月的账单",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗已经分期的可以一次还款吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗是不是可以随时还",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么蚂蚁借呗里借多久的选项中只有六个月",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n借呗借款分期后到期还可以再分期还款吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n银行卡支付了什么花呗再重复扣款",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗额度怎么看",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n网商贷的利息跟借呗利息差多少",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n怎么现在借呗只能***个月还款",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我逾期还款已还还能用花呗吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n借呗可以分期还",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我这个月还完花呗了，怎么还要还",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n查看花呗收款码",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗审核没通过，好何重新提交",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗账单分期提起还款还有手续费么",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么，我的没有花呗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我的花呗是负的有影响吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n没有银行卡可以还花呗嘛",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我已经还完款了，可是这是淘宝退款自动退到花呗里的，可是在他退之前我已经先还完了这个款，他这个钱退去哪里了",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n付款的地方没有花呗支付",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗自动还款余额宝代扣功能怎么开",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么我的蚂蚁借呗额度突然降了那么多",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗为什么提前还款了之后没有额度",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n你们家支持花呗支付么",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗不用额度是变吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗可以迟一个星期还吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n?借呗上传身份证",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗如果我开通了不用会有费用吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n提前还款的话利息会少么",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我的什么时候才能用花呗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么我没有花呗可以用",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n借呗是要全部还清才可以再次借款吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗我已经还了，为什么现在无法使用",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n在哪看用了几次花呗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗能不能换手机号码呀",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗今天的还款记录",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n退到花呗里的钱怎么办",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我能把我另一个关联账号花呗转到现在用的账号吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n以前能用花呗缴费着嘛",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n扫手机上的收钱码怎么不支持花呗收钱",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n说是退回了花呗但没看到钱进了哪儿了",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗全部还了 能不能关闭",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗我能接多少钱",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗还钱要手机验证码，我手机号码没有再用，所以收不到信息，应该怎么还",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么花呗无法买车票",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗钱已经还上了怎么还没有恢复额度",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n申请借呗，多久能审核额度",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n借呗开通不了",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗额度为什么会降下",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n借呗还款有短信通知吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我的借呗额度什么时候恢复，我希望快点",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗号和支付宝账号不一样",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁花呗是可以自己设置还款时间吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么商家开通了花呗服务 我却无法用花呗付款",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n假如现在手头有点紧，***号还不了花呗，如果选择分期这个月底前一次还清还收取额外费用吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n为什么我的蚂蚁借呗突然封了",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n未买东西怎么自动从花呗里扣款",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗逾期一两天会影响什么？利息是多少",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n借呗借款期限只有***个月和***个月",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗借呗都不见了",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n不想要网商贷怎么变回借呗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n如何查看花呗还款的具体",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗借***一个月还多少",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n要还花呗的钱，怎么操作，昨晚用的",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我有一个***天免息的借呗券",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗充加油卡吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n花呗额度与芝麻分有关系吗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n嗨为什么不借呗提现了",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n您可登录支付宝号使用花呗",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n您说的是您登录不上您开通花呗的 支付宝账户",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗怎么每期提前还",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n蚂蚁借呗利率怎么计算",
    },
    {
        "role": "user",
        "content": "Translate the following segment into Spanish, without additional explanation.\n\n我怎么才能提高借呗金额",
    },
]
prompt = llm.get_tokenizer().apply_chat_template(
    prompt_batch,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="np",
)


# Generate texts from the prompts.
outputs = llm.generate(prompt, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
