from __future__ import annotations
import spacy
from pathlib import Path
import json
import uuid
from typing import Iterable, List, Dict, Any, Optional
from PIL import Image

DEFAULT_CACHE_DIR = "/mnt/nvme0/tdy/cache_datasets"
DEFAULT_NUM_PROCS = 1024

# English model; extend as needed
_nlp_en = spacy.load("en_core_web_sm")


def dataset_to_jsonl(
    ds,
    output_path: str | Path,
    field_map: dict[str, str],
    extra: dict[str, object] | None = None,
) -> None:
    """Export a dataset to jsonl with simple field mapping.

    field_map maps output_field -> input_field.
    extra adds fixed key/value pairs to every record.
    """
    extra = extra or {}
    records = []
    for row in ds:
        rec = {out_k: row[in_k] for out_k, in_k in field_map.items()}
        rec.update(extra)
        records.append(rec)
    write_jsonl(records, output_path)


def split_text_by_sentence(text: str, lang: str = "en") -> dict[int, str]:
    """
    Use SpaCy to split text into sentences. Returns {chunk_id: chunk_text}.
    """
    if lang == "en":
        doc = _nlp_en(text)
    else:
        raise NotImplementedError("Only English is supported; load zh model if needed.")

    chunks: dict[int, str] = {}
    for i, sent in enumerate(doc.sents):
        s = sent.text.strip()
        if s:
            chunks[i] = s
    return chunks


def find_evidence_chunks(
    evidence: str | list[str], chunks: dict[int, str]
) -> list[int]:
    """Find sentence indices that contain evidence strings."""
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


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory of the given path exists and return a Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str | Path) -> None:
    """Write a sequence of records to a JSONL file."""
    output_path = ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + " ")


def collect_records(dataset, field: str = "records") -> List[Dict[str, Any]]:
    """Flatten and collect the list of records from the given field."""
    items: List[Dict[str, Any]] = []
    for rec_list in dataset[field]:
        if not rec_list:
            continue
        items.extend(rec_list)
    return items


def save_mapped_records_jsonl(
    dataset, output_path: str | Path, field: str = "records"
) -> None:
    """Collect records and write them to a JSONL file."""
    records = collect_records(dataset, field=field)
    write_jsonl(records, output_path)


def crop_image(image_path: str, box: list, save_dir: str = "./tmp") -> Optional[str]:
    """Crop an image by the given box and save it, returning the saved path."""
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"can't open the image: {image_path} error: {e}")
    x, y, w, h = box
    img_w, img_h = img.size
    if not (0 <= x < img_w and 0 <= y < img_h):
        return None
    right = min(x + w, img_w)
    lower = min(y + h, img_h)
    crop_box = (x, y, right, lower)
    cropped = img.crop(crop_box)
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = save_dir_path / filename
    cropped.save(str(save_path), quality=95)
    return str(save_path)
