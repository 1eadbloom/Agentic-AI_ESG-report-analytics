"""
search_agent.py — 資料搜尋與擷取代理（Search Agent）

功能：
  1. 掃描 data/reports/ 中所有 PDF / TXT
  2. 以 PyMuPDF 提取文字（數字能留在原句；失敗時退回 pdfplumber、pypdf）
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
    REPORTS_DIR, PROCESSED_DIR, KB_DIR, clean_text, classify_pillar,
    get_collection, get_embedding_fn, EmbeddingFn,
)

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
HASH_FILE     = PROCESSED_DIR / ".file_hashes.json"

# （僅 pdfplumber 退路使用）是否把偵測到的表格另外轉成「欄位 | 欄位」文字附在頁面後面
# （表格內容會與一般文字重複，chunk 數量會增加；先用 inspect_pdf_page.py 確認有需要再開）
EXTRACT_TABLES = False

# 頁碼標記：用控制字元，不會出現在正常文字，也不會被章節標題規則誤判
_PAGE_MARK_RE = re.compile(r"\x00P(\d+)\x00\n?")


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
    return json.loads(HASH_FILE.read_text(encoding="utf-8")) if HASH_FILE.exists() else {}

def save_hashes(h: dict):
    HASH_FILE.write_text(json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8")

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── PDF 提取 ──────────────────────────────────────────────────────
def _extract_pdf_pymupdf(path: Path) -> list[dict]:
    """PyMuPDF 擷取。對這類多欄位資訊圖表式排版，數字能留在原句中
    （例：「新臺幣0.6億，占整體投資金額約36%」），pdfplumber 則會把數字拆到相鄰行。"""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz   # 舊版套件名稱
    pages = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, 1):
            t = page.get_text("text") or ""
            if t.strip():
                pages.append({"page": i, "text": t})
    return pages


def extract_pdf(path: Path) -> list[dict]:
    """每頁提取文字，回傳 [{page, text}, ...]。優先 PyMuPDF，失敗再退回 pdfplumber、pypdf。"""
    try:
        pages = _extract_pdf_pymupdf(path)
        if pages:
            return pages
    except Exception as e:
        print(f"  ⚠️  PyMuPDF 失敗（{e}），改用 pdfplumber（數字可能被拆離原句；建議 pip install pymupdf）")
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                t = _extract_page_text(page)
                if t.strip():
                    pages.append({"page": i, "text": t})
        return pages
    except Exception as e:
        print(f"  ⚠️  pdfplumber 失敗，改用 pypdf：{e}")
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [{"page": i+1, "text": p.extract_text() or ""}
                for i, p in enumerate(reader.pages) if p.extract_text()]


def _is_page_no(line: str, page_no: int) -> bool:
    """獨立一行的數字，且與 PDF 頁序相差不大（印刷頁碼通常等於或略小於頁序），才視為頁碼。"""
    t = line.strip()
    return t.isdigit() and len(t) <= 4 and abs(int(t) - page_no) <= 5


def _clean_page(text: str, page_no: int = 0) -> str:
    """頁面文字清理。

    與 esg_utils.clean_text 的差別：**不再刪除「只有數字的獨立一行」**。
    永續報告書的表格常把每個儲存格擷取成獨立一行（例如 168、2024、36），
    原本的規則會把這些數字當成頁碼一併刪掉。這裡只移除頁首/頁尾、且數值接近頁序的頁碼。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"2024 Sustainability Report\s*", "", text)
    text = re.sub(r"Sustainability Report\s*", "", text)
    lines = text.split("\n")
    if lines and _is_page_no(lines[0], page_no):
        lines.pop(0)
    if lines and _is_page_no(lines[-1], page_no):
        lines.pop()
    text = "\n".join(lines)
    text = re.sub(r"(?m)^[\s\(\):]+$", "", text)     # 只剩括號、冒號的殘渣行
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_page_text(page) -> str:
    """單頁文字；EXTRACT_TABLES 開啟時，額外附上表格（含表格外的「單位」註記）。"""
    text = page.extract_text() or ""
    if not EXTRACT_TABLES:
        return text
    try:
        tables = page.extract_tables()
    except Exception:
        tables = []
    units = [ln.strip() for ln in text.split("\n") if "單位" in ln][:2]
    blocks = []
    for t in tables:
        rows = [" | ".join((c or "").replace("\n", " ").strip() for c in row)
                for row in t if any(c for c in row)]
        if len(rows) >= 2:
            note = f"（{'；'.join(units)}）" if units else ""
            blocks.append("【表格】" + note + "\n" + "\n".join(rows))
    return text + ("\n\n" + "\n\n".join(blocks) if blocks else "")


def extract_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"page": 1, "text": text}]


# ── 章節切分 ─────────────────────────────────────────────────────
# 識別常見中文 ESG 報告的章節標題格式
_SECTION_RE = re.compile(
    r"(?m)^("
    r"\d+[\. \t　][^\n]{3,40}|"        # 1. 章節名稱（分隔符不含換行，避免「數字儲存格＋下一行文字」被誤判為標題）
    r"[一二三四五六七八九十]+[、\. \t　][^\n]{3,30}|"  # 一、章節
    r"(?:環境|社會|治理|永續|前言|附錄)[^\n]{0,30}"  # 關鍵字開頭
    r")$"
)

def _strip_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    """移除頁碼標記，並回傳 [(標記在「乾淨文字」中的位置, 頁碼), ...]。"""
    out, offs, pos, last = [], [], 0, 0
    for m in _PAGE_MARK_RE.finditer(text):
        out.append(text[last:m.start()])
        pos += m.start() - last
        offs.append((pos, int(m.group(1))))
        last = m.end()
    out.append(text[last:])
    return "".join(out), offs


def _page_at(offs: list[tuple[int, int]], start_page: int, pos: int) -> int:
    page = start_page
    for off, pg in offs:
        if off <= pos:
            page = pg
        else:
            break
    return page


def split_into_sections(pages: list[dict], source_file: str,
                        company: str = "", report_year: str = "",
                        doc_type: str = "report") -> list[ESGSection]:
    """合併全文後做章節切分，每個 section 進一步切成 chunks（每個 chunk 帶真實頁碼）。"""
    full_text = "\n\n".join(
        f"\x00P{p['page']}\x00\n" + _clean_page(p["text"], p["page"]) for p in pages)
    cur_page = min(p["page"] for p in pages)

    splits = _SECTION_RE.split(full_text)
    sections: list[ESGSection] = []
    sec_id = 0

    # splits 格式：[前文, 標題1, 內容1, 標題2, 內容2, ...]
    i = 0
    while i < len(splits):
        raw = splits[i]
        clean, offs = _strip_with_offsets(raw)
        start_page = cur_page
        if offs:
            cur_page = offs[-1][1]            # 無論之後是否略過，頁碼都要往前推進
        lead = len(clean) - len(clean.lstrip())
        offs = [(max(0, o - lead), pg) for o, pg in offs]
        chunk_text_raw = clean.strip()
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
            c_clean, c_offs = _strip_with_offsets(splits[i + 1])
            c_start = cur_page
            if c_offs:
                cur_page = c_offs[-1][1]
            c_lead = len(c_clean) - len(c_clean.lstrip())
            c_offs = [(max(0, o - c_lead), pg) for o, pg in c_offs]
            content, content_offs, content_start = c_clean.strip(), c_offs, c_start
            i += 2
        else:
            title = f"段落{sec_id}"
            content, content_offs, content_start = chunk_text_raw, offs, start_page
            i += 1

        if not content:
            continue

        for chunk in _chunk(content, title, sec_id, source_file,
                            content_offs, content_start,
                            company, report_year, doc_type):
            sections.append(chunk)
            sec_id += 1

    return sections


def _chunk(text: str, title: str, base_id: int,
           source_file: str, offs: list[tuple[int, int]], start_page: int,
           company: str = "", report_year: str = "",
           doc_type: str = "report") -> list[ESGSection]:
    def _range(a: int, b: int) -> str:
        p1 = _page_at(offs, start_page, a)
        p2 = _page_at(offs, start_page, max(a, b - 1))
        return f"p{p1}" if p1 == p2 else f"p{p1}-{p2}"

    if len(text) <= CHUNK_SIZE:
        return [ESGSection(
            section_id=f"{source_file}::{base_id}",
            title=title, text=text,
            pillar=classify_pillar(title + " " + text),
            source_file=source_file,
            page_range=_range(0, len(text)),
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
            page_range=_range(start, min(end, len(text))),
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
    ), encoding="utf-8") 

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
        targets = list(files)
        if include_kb:   # 原本傳入 files 時知識庫會被直接略過，導致最佳實踐永遠查不到資料
            targets += sorted(KB_DIR.glob("*.md")) + sorted(KB_DIR.glob("*.txt"))
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
                            company="知識庫" if _doc_type == "knowledge_base" else company,
                            doc_type=_doc_type)
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
