import argparse
import json
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


PEERQA_ZIP_URL = (
    "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/"
    "tudatalib/4467/peerqa-data-v1.0.zip?sequence=5&isAllowed=y"
)


def find_data_files(base_path: Path) -> tuple[Path | None, Path | None]:
    # 优先在根目录和常见子目录里找
    candidates = [base_path, base_path / "data", base_path / "raw", base_path / "peerqa-data-v1.0"]
    for folder in candidates:
        qa_path = folder / "qa.jsonl"
        papers_path = folder / "papers.jsonl"
        if qa_path.exists() and papers_path.exists():
            return qa_path, papers_path

    # 兜底：递归搜索
    qa_path = None
    papers_path = None
    for p in base_path.rglob("qa.jsonl"):
        qa_path = p
        break
    for p in base_path.rglob("papers.jsonl"):
        papers_path = p
        break
    return qa_path, papers_path


def maybe_download_peerqa(base_path: Path) -> tuple[Path, Path]:
    qa_path, papers_path = find_data_files(base_path)
    if qa_path and papers_path:
        return qa_path, papers_path

    print("ℹ️ 未找到 qa.jsonl / papers.jsonl，开始下载 peerqa-data-v1.0.zip ...")
    base_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmpf:
        tmp_zip = Path(tmpf.name)
    urllib.request.urlretrieve(PEERQA_ZIP_URL, tmp_zip)

    extract_dir = base_path / "data"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(extract_dir)
    tmp_zip.unlink(missing_ok=True)

    qa_path, papers_path = find_data_files(base_path)
    if not (qa_path and papers_path):
        raise FileNotFoundError("下载后仍未找到 qa.jsonl / papers.jsonl")
    return qa_path, papers_path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "--p", type=str, required=True)
    parser.add_argument(
        "--download_if_missing",
        action="store_true",
        help="若本地没有 qa.jsonl/papers.jsonl，则自动下载官方 zip",
    )
    args = parser.parse_args()

    base_path = Path(args.path)

    if args.download_if_missing:
        qa_path, papers_path = maybe_download_peerqa(base_path)
    else:
        qa_path, papers_path = find_data_files(base_path)
        if not (qa_path and papers_path):
            raise FileNotFoundError(
                "未找到 qa.jsonl / papers.jsonl。可加 --download_if_missing 自动下载。"
            )

    # 读取 papers，并按 paragraph 聚合，构造 corpus
    # chunk_id 格式: {paper_id}__p{pidx}
    paper_para_texts: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for row in load_jsonl(papers_path):
        paper_id = str(row["paper_id"])
        pidx = int(row["pidx"])
        sidx = int(row.get("sidx", -1))
        content = str(row.get("content", "")).strip()
        if content:
            paper_para_texts[(paper_id, pidx)].append((sidx, content))

    corpus_rows: list[dict] = []
    for (paper_id, pidx), sents in sorted(paper_para_texts.items()):
        sents = sorted(sents, key=lambda x: x[0])
        text = " ".join(sent for _, sent in sents).strip()
        chunk_id = f"{paper_id}__p{pidx}"
        corpus_rows.append({"id": chunk_id, "title": paper_id, "text": text})

    # 用 papers 的 idx -> pidx 映射，把 QA 的 evidence idx 映射到 paragraph
    idx_to_pidx: dict[tuple[str, int], int] = {}
    for row in load_jsonl(papers_path):
        idx_to_pidx[(str(row["paper_id"]), int(row["idx"]))] = int(row["pidx"])

    # queries / qrels
    queries_rows: list[dict] = []
    qrels_rows: list[dict] = []
    query_id = 0
    for qa in load_jsonl(qa_path):
        question_text = str(qa.get("question", "")).strip()
        paper_id = str(qa.get("paper_id", ""))
        evidence = qa.get("answer_evidence_mapped")

        # answer_evidence_mapped 可能为 None
        pidx_set: set[int] = set()
        if isinstance(evidence, list):
            for ev in evidence:
                for idx in ev.get("idx", []):
                    if idx is None:
                        continue
                    key = (paper_id, int(idx))
                    if key in idx_to_pidx:
                        pidx_set.add(idx_to_pidx[key])

        chunk_ids = [f"{paper_id}__p{pidx}" for pidx in sorted(pidx_set)]

        queries_rows.append({"id": query_id, "text": question_text})
        qrels_rows.append({"chunk_ids": chunk_ids, "extra": 1, "query_id": query_id})
        query_id += 1

    out_dir = base_path / "test"
    write_jsonl(corpus_rows, out_dir / "corpus.jsonl")
    write_jsonl(queries_rows, out_dir / "queries.jsonl")
    write_jsonl(qrels_rows, out_dir / "qrels.jsonl")

    print(f"✅ corpus.jsonl saved to {out_dir / 'corpus.jsonl'}")
    print(f"✅ queries.jsonl saved to {out_dir / 'queries.jsonl'}")
    print(f"✅ qrels.jsonl saved to {out_dir / 'qrels.jsonl'}")


if __name__ == "__main__":
    main()
    print("✅ done")
