import re
from typing import Dict, List, Union
import spacy
from typing import Dict

# 加载 SpaCy 模型
# 英文模型
nlp_en = spacy.load("en_core_web_sm")
# 中文模型，如果处理中文文本，可以安装 zh_core_web_sm
# nlp_zh = spacy.load("zh_core_web_sm")


def split_text_by_sentence(text: str, lang: str = "en") -> Dict[int, str]:
    """
    使用 SpaCy 按句子分割文本，返回 {chunk_id: chunk_text}

    Args:
        text: 待分割长文本
        lang: "en" 或 "zh"，指定语言

    Returns:
        chunks: dict，key 为 chunk_id，value 为句子文本
    """
    if lang == "en":
        doc = nlp_en(text)
    else:
        raise NotImplementedError("当前仅支持英文，如果需要中文请加载中文模型")

    chunks = {}
    for i, sent in enumerate(doc.sents):
        s = sent.text.strip()
        if s:
            chunks[i] = s

    return chunks


def find_evidence_chunks(
    evidence: Union[str, List[str]], chunks: Dict[int, str]
) -> List[int]:
    """
    找出 evidence 出现在 chunks 中的索引

    Args:
        evidence: 单个字符串或字符串列表
        chunks: {chunk_id: chunk_text}

    Returns:
        list of chunk_id
    """
    if isinstance(evidence, str):
        evidence_list = [evidence]
    else:
        evidence_list = evidence

    matched_ids = set()

    for e in evidence_list:
        e = e.strip()
        if not e:
            continue
        for idx, text in chunks.items():
            if e in text:
                matched_ids.add(idx)

    return sorted(matched_ids)
