"""
search_agent.py — 資料搜尋與擷取代理（Search Agent）

功能：
  1. 掃描 data/reports/ 中所有 PDF / TXT
  2. 以 pdfplumber 提取文字（中文 ESG 報告效果優於 pypdf）
  3. 清理 + 段落切分
  4. Embed → upsert 進 ChromaDB（冪等，hash 判斷是否重建）
  5. 回傳結構化的 sections 清單供後續 Agent 使用

用法：
  python search_agent.py                # 增量更新
  python search_agent.py --rebuild      # 全量重建
  python search_agent.py --file xxx.pdf # 只處理單一檔案
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from esg_utils import (
    REPORTS_DIR, PROCESSED_DIR, clean_text, classify_pillar,
    get_collection, get_embedding_fn, EmbeddingFn,
)

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
HASH_FILE     = PROCESSED_DIR / ".file_hashes.json"


# ── 資料結構 ──────────────────────────────────────────────────────
@dataclass
class ESGSection:
    section_id: str
    title: str
    text: str
    pillar: str           # E / S / G / O
    source_file: str
    page_range: str = ""  # e.g. "p12-15"
    company: str = ""     # 公司名稱，由 process_file 傳入
    report_year: str = "" # 報告年度，從檔名推斷
    doc_type: str = "report"  # "report" 或 "knowledge_base"


# ── Hash 管理（冪等更新）─────────────────────────────────────────
def load_hashes() -> dict:
    return json.loads(HASH_FILE.read_text()) if HASH_FILE.exists() else {}

def save_hashes(h: dict): HASH_FILE.write_text(json.dumps(h, indent=2))

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── PDF 提取 ──────────────────────────────────────────────────────
def extract_pdf(path: Path) -> list[dict]:
    """每頁提取文字，回傳 [{page, text}, ...]"""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                t = page.extract_text() or ""
                if t.strip():
                    pages.append({"page": i, "text": t})
        return pages
    except Exception as e:
        print(f"  ⚠️  pdfplumber 失敗，改用 pypdf：{e}")
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [{"page": i+1, "text": p.extract_text() or ""}
                for i, p in enumerate(reader.pages) if p.extract_text()]


def extract_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"page": 1, "text": text}]


# ── 章節切分 ─────────────────────────────────────────────────────
# 識別常見中文 ESG 報告的章節標題格式
_SECTION_RE = re.compile(
    r"(?m)^("
    r"\d+[\.\s][^\n]{3,40}|"          # 1. 章節名稱
    r"[一二三四五六七八九十]+[、\.\s][^\n]{3,30}|"  # 一、章節
    r"(?:環境|社會|治理|永續|前言|附錄)[^\n]{0,30}"  # 關鍵字開頭
    r")$"
)

def split_into_sections(pages: list[dict], source_file: str,
                        company: str = "", report_year: str = "",
                        doc_type: str = "report") -> list[ESGSection]:
    """合併全文後做章節切分，每個 section 進一步切成 chunks。"""
    full_text = "\n\n".join(clean_text(p["text"]) for p in pages)
    page_nums = [p["page"] for p in pages]
    p_min, p_max = min(page_nums), max(page_nums)

    # 以章節標題切分
    splits = _SECTION_RE.split(full_text)
    sections: list[ESGSection] = []
    sec_id = 0

    # splits 格式：[前文, 標題1, 內容1, 標題2, 內容2, ...]
    i = 0
    while i < len(splits):
        chunk_text_raw = splits[i].strip()
        if not chunk_text_raw:
            i += 1
            continue
        # 跳過以亂碼為主的段落（中文字比率過低）
        chinese_chars = sum(1 for c in chunk_text_raw if "\u4e00" <= c <= "\u9fff")
        if len(chunk_text_raw) > 50 and chinese_chars / max(len(chunk_text_raw), 1) < 0.05:
            i += 1
            continue
        # 判斷是否為標題（短且沒有換行）
        is_title = len(chunk_text_raw) < 60 and "\n" not in chunk_text_raw
        if is_title and i + 1 < len(splits):
            title = chunk_text_raw
            content = splits[i + 1].strip()
            i += 2
        else:
            title = f"段落{sec_id}"
            content = chunk_text_raw
            i += 1

        if not content:
            continue

        # 對超長內容做 chunking
        for chunk in _chunk(content, title, sec_id, source_file, p_min, p_max, company, report_year, doc_type):
            sections.append(chunk)
            sec_id += 1

    return sections


def _chunk(text: str, title: str, base_id: int,
           source_file: str, p_min: int, p_max: int,
           company: str = "", report_year: str = "",
           doc_type: str = "report") -> list[ESGSection]:
    if len(text) <= CHUNK_SIZE:
        return [ESGSection(
            section_id=f"{source_file}::{base_id}",
            title=title, text=text,
            pillar=classify_pillar(title + " " + text),
            source_file=source_file,
            page_range=f"p{p_min}-{p_max}",
            company=company, report_year=report_year, doc_type=doc_type,
        )]
    chunks = []
    start = 0
    sub = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        piece = text[start:end]
        chunks.append(ESGSection(
            section_id=f"{source_file}::{base_id}_{sub}",
            title=title, text=piece.strip(),
            pillar=classify_pillar(title + " " + piece),
            source_file=source_file,
            page_range=f"p{p_min}-{p_max}",
            company=company, report_year=report_year, doc_type=doc_type,
        ))
        sub += 1
        start = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > start else end
    return chunks


# ── 主流程 ────────────────────────────────────────────────────────
def infer_meta(path: Path) -> tuple[str, str]:
    """從檔名推斷公司名稱和報告年度。
    例：2024年永續報告書_中_中信證券000616.pdf → ('中信證券', '2024')
    """
    stem = path.stem
    # 嘗試取出年份
    year_m = re.search(r"(20[0-9]{2})", stem)
    year = year_m.group(1) if year_m else "unknown"
    # 移除年份、常見關鍵字、數字編號，剩下的視為公司名稱
    name = re.sub(r"20[0-9]{2}|永續報告書?|ESG|Sustainability|Report|_中_|_EN_|[_\-\s]+", " ", stem)
    name = re.sub(r"[0-9]{6,}", "", name).strip()
    if not name or len(name) < 2:
        name = stem[:10]
    return name, year


def process_file(path: Path, ef: EmbeddingFn, collection,
                 force: bool = False, company: str = "",
                 doc_type: str = "report") -> list[ESGSection]:
    hashes = load_hashes()
    h = file_hash(path)
    rel = path.name

    if not force and hashes.get(rel) == h:
        print(f"  ⏭  未變動，略過：{rel}")
        return []

    print(f"  📄 處理：{rel}")
    if path.suffix.lower() == ".pdf":
        pages = extract_pdf(path)
    else:
        pages = extract_txt(path)

    if not pages:
        print(f"  ⚠️  無法提取文字：{rel}")
        return []

    inferred_company, report_year = infer_meta(path)
    _company = company or inferred_company
    sections = split_into_sections(pages, rel,
                                   company=_company,
                                   report_year=report_year,
                                   doc_type=doc_type)
    print(f"     → {len(sections)} 個 chunks，"
          f"E:{sum(1 for s in sections if s.pillar=='E')} "
          f"S:{sum(1 for s in sections if s.pillar=='S')} "
          f"G:{sum(1 for s in sections if s.pillar=='G')} "
          f"O:{sum(1 for s in sections if s.pillar=='O')}")

    # Upsert into ChromaDB
    batch = 64
    for i in range(0, len(sections), batch):
        b = sections[i:i+batch]
        collection.upsert(
            ids       =[s.section_id for s in b],
            documents =[s.text for s in b],
            metadatas =[{
                "title": s.title, "pillar": s.pillar,
                "source_file": s.source_file, "page_range": s.page_range,
                "company": s.company, "report_year": s.report_year,
                "doc_type": s.doc_type,
            } for s in b],
        )

    # 儲存 processed JSON
    out = PROCESSED_DIR / (path.stem + ".json")
    out.write_text(json.dumps(
        [asdict(s) for s in sections], ensure_ascii=False, indent=2
    ))

    hashes[rel] = h
    save_hashes(hashes)
    return sections


def run_search_agent(files: list[Path] | None = None,
                     rebuild: bool = False,
                     company: str = "",
                     include_kb: bool = True) -> list[ESGSection]:
    print("\n🔍 Search Agent 啟動")
    ef = get_embedding_fn()
    client, collection = get_collection(ef)

    if rebuild:
        print("  🗑  清空 ChromaDB collection")
        try: client.delete_collection("esg_reports")
        except Exception: pass
        client, collection = get_collection(ef)
        save_hashes({})

    if files:
        targets = files
    else:
        targets = sorted(REPORTS_DIR.glob("*.pdf")) + sorted(REPORTS_DIR.glob("*.txt"))
        if include_kb:
            targets += sorted(KB_DIR.glob("*.md")) + sorted(KB_DIR.glob("*.txt"))
    if not targets:
        print(f"  ⚠️  {REPORTS_DIR} 中沒有找到 PDF/TXT 檔案")
        return []

    all_sections: list[ESGSection] = []
    for p in targets:
        _doc_type = "knowledge_base" if str(p).startswith(str(KB_DIR)) else "report"
        secs = process_file(p, ef, collection, force=rebuild,
                            company=company, doc_type=_doc_type)
        all_sections.extend(secs)

    total = collection.count()
    print(f"\n✅ Search Agent 完成：共 {total} 個 chunks 已建立索引\n")
    return all_sections


# ── CLI 進入點 ────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ESG Search Agent：PDF 解析與向量索引")
    ap.add_argument("--rebuild", action="store_true", help="全量重建 ChromaDB")
    ap.add_argument("--file",    type=str, default=None, help="只處理單一檔案（檔名）")
    args = ap.parse_args()

    targets = None
    if args.file:
        p = REPORTS_DIR / args.file
        if not p.exists():
            print(f"❌ 找不到檔案：{p}")
            sys.exit(1)
        targets = [p]

    run_search_agent(files=targets, rebuild=args.rebuild)
