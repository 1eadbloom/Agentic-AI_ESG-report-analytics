"""
citation.py — 引用來源追蹤（Citation Source Tracking）

新增實體與確定性歸因邏輯（research R1/R2/R3）：
  - Citation：連結 AI 結論與原始文件片段
  - AnalysisItem：結論文字 + 對應 citations
  - 確定性歸因：embedding 餘弦相似度 top-k（threshold 以上），失敗時退到關鍵字重疊
  - 三層 metadata 隔離守門（FR-004 / SC-002）

本模組與 LangGraph 完全解耦，僅依賴 esg_utils.get_embedding_fn()，
因此可單獨做單元測試（research R5）。
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from esg_utils import get_embedding_fn

# ── 常數 ──────────────────────────────────────────────────────────
DEFAULT_THRESHOLD      = 0.30   # 相似度門檻，低於此標記 unattributed（FR-006）
DEFAULT_TOP_K          = 3      # 最多列出的來源片段數（FR-005）
EXCERPT_MAX_CHARS      = 100    # 摘錄上限（FR-008）
DISPLAY_EXCERPT_CHARS  = 60     # Markdown 顯示時截斷長度
VALID_DOC_TYPES        = ("report", "knowledge_base")
UNATTRIBUTED           = "unattributed"
UNATTRIBUTED_NOTE      = "非文件依據 · 通用知識或經驗判斷，非特定文件片段"


# ── 資料結構 ──────────────────────────────────────────────────────
@dataclass
class Citation:
    """單一來源引用（欄位對應 data-model.md）。"""
    company: str
    report_year: str
    source_file: str
    chunk_id: str
    page_range: str
    doc_type: str
    excerpt: str = ""

    @property
    def is_unattributed(self) -> bool:
        return self.chunk_id == UNATTRIBUTED

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Citation":
        return cls(**d)


@dataclass
class AnalysisItem:
    """包裝單一 LLM 結論與其證據（data-model.md）。"""
    text: str
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text,
                "sources": [_as_source(c) for c in self.citations]}

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisItem":
        return cls(text=d.get("text", ""),
                   citations=[Citation.from_dict(c) for c in d.get("sources", [])])


# ── 建構輔助 ──────────────────────────────────────────────────────
def _as_source(obj: Any) -> dict:
    return obj.to_dict() if isinstance(obj, Citation) else obj


def make_citation(section: Any, excerpt: str = "") -> Citation:
    """從既有的 Document Fragment（section 物件）衍生 Citation（唯讀現有欄位）。"""
    return Citation(
        company=str(getattr(section, "company", "")),
        report_year=str(getattr(section, "report_year", "")),
        source_file=str(getattr(section, "source_file", "")),
        chunk_id=str(getattr(section, "section_id", "")),
        page_range=str(getattr(section, "page_range", "")),
        doc_type=str(getattr(section, "doc_type", "report")),
        excerpt=excerpt,
    )


def make_unattributed(run_company: str, run_year: str,
                      note: str = UNATTRIBUTED_NOTE) -> Citation:
    """FR-006：不可溯源的結論標記 unattributed，並附說明文字。"""
    return Citation(
        company=run_company,
        report_year=run_year,
        source_file="",
        chunk_id=UNATTRIBUTED,
        page_range="",
        doc_type="",
        excerpt=note,
    )


# ── 隔離守門（FR-004 / SC-002）───────────────────────────────────
def assert_section_scope(section: Any, run_company: str,
                         run_year: str = "") -> None:
    """斷言 section 的 metadata 符合 run scope；不符即拋出 ValueError。"""
    sid = getattr(section, "section_id", "?")
    s_company = str(getattr(section, "company", "") or "")
    s_year = str(getattr(section, "report_year", "") or "")
    s_type = str(getattr(section, "doc_type", "report") or "")

    if run_company and s_company != run_company:
        raise ValueError(
            f"跨公司隔離違規：section {sid} 的公司 {s_company!r} != 本次執行公司 {run_company!r}")
    if run_year and s_year != run_year:
        raise ValueError(
            f"跨年度隔離違規：section {sid} 的年度 {s_year!r} != 本次執行年度 {run_year!r}")
    if s_type not in VALID_DOC_TYPES:
        raise ValueError(
            f"doc_type 無效：section {sid} 的 doc_type = {s_type!r}")


def _in_scope(section: Any, run_company: str, run_year: str) -> bool:
    try:
        assert_section_scope(section, run_company, run_year)
        return True
    except ValueError:
        return False


def is_in_scope(section: Any, run_company: str, run_year: str = "") -> bool:
    """公開版 scope 檢查（供 interpretation agent 過濾候選片段）。"""
    return _in_scope(section, run_company, run_year)


# ── 相似度評分（R1）───────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = n_a = n_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        n_a += x * x
        n_b += y * y
    if n_a == 0.0 or n_b == 0.0:
        return 0.0
    return dot / math.sqrt(n_a * n_b)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def keyword_overlap(text: str, section_text: str) -> float:
    """退路評分：結論 token 中來自該 section 的比例。"""
    ta = set(_TOKEN_RE.findall(text.lower()))
    tb = set(_TOKEN_RE.findall(section_text.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def score_against_sections(text: str, sections: list) -> list[tuple[Any, float]]:
    """回傳 [(section, score)]，依分數由高至低排序。

    優先使用 embedding 餘弦相似度（沿用既有已載入模型）；embedding 不可用時
    退回關鍵字重疊。不會額外呼叫 LLM（SC-004）。
    """
    sections = [s for s in sections if getattr(s, "text", "")]
    if not sections:
        return []
    try:
        ef = get_embedding_fn()
        vecs = ef([text] + [getattr(s, "text", "") for s in sections])
        base = vecs[0]
        scored = [(s, cosine_similarity(base, v))
                  for s, v in zip(sections, vecs[1:])]
    except Exception:
        scored = [(s, keyword_overlap(text, getattr(s, "text", "")))
                  for s in sections]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


# ── 來源選取（R1 + FR-005）────────────────────────────────────────
def select_sources(text: str, sections: list, run_company: str,
                   run_year: str = "", top_k: int = DEFAULT_TOP_K,
                   threshold: float = DEFAULT_THRESHOLD) -> list[Citation]:
    """將一則結論確定性地歸因給候選 section。

    - 只接受 run scope 內、doc_type == "report" 的片段（KB 不混入公司結論）
    - 取 similarity >= threshold 的前 top_k 名作為 citations
    - 沒有任何合格片段時標記 unattributed（FR-006）
    """
    candidates = [
        s for s in sections
        if _in_scope(s, run_company, run_year)
        and getattr(s, "doc_type", "report") == "report"
    ]
    if not candidates:
        return [make_unattributed(run_company, run_year)]

    scored = score_against_sections(text, candidates)
    picks = [s for s, sc in scored if sc >= threshold][:top_k]
    if not picks:
        return [make_unattributed(run_company, run_year)]

    return [make_citation(s, excerpt=truncate(getattr(s, "text", "")))
            for s in picks]


# ── 摘錄輔助（FR-008）────────────────────────────────────────────
def truncate(text: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    t = " ".join((text or "").split())
    return t[:max_chars]


def display_excerpt(text: str, max_chars: int = DISPLAY_EXCERPT_CHARS) -> str:
    """Markdown 顯示用摘錄：超過長度以 … 結尾。"""
    t = truncate(text, max_chars)
    if len((text or "")) > max_chars:
        t += "…"
    return t


# ── Markdown 渲染（contracts/markdown-report.md）──────────────────
def format_source_part(source: Any, include_excerpt: bool = False) -> str:
    c = source if isinstance(source, Citation) else Citation.from_dict(source)
    label = "KB 來源" if c.doc_type == "knowledge_base" else "來源"
    core = (f"{label}: {c.company} · {c.report_year} · "
            f"chunk {c.chunk_id} · {c.page_range}")
    if include_excerpt and c.excerpt:
        core += f"：「{display_excerpt(c.excerpt)}」"
    return core


def citations_to_markdown(citations: list, display_excerpt: bool = True) -> str:
    """將 citations 轉成內嵌 parenthetical（含前導空白）。"""
    citations = list(citations or [])
    if not citations:
        return ""
    if len(citations) == 1 and getattr(citations[0], "chunk_id", "") == UNATTRIBUTED:
        note = getattr(citations[0], "excerpt", "") or UNATTRIBUTED_NOTE
        return f" ({note})"
    parts = [
        format_source_part(c, include_excerpt=(display_excerpt and i == 0))
        for i, c in enumerate(citations)
    ]
    return " (" + " ; ".join(parts) + ")"


def union_citations(citation_lists: list[list]) -> list[Citation]:
    """多個清單的合併（保序去重），供策略繼承弱點引用使用。"""
    seen = set()
    merged: list[Citation] = []
    for lst in citation_lists:
        for c in lst:
            c = c if isinstance(c, Citation) else Citation.from_dict(c)
            key = (c.company, c.report_year, c.source_file,
                   c.chunk_id, c.page_range, c.doc_type)
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
    return merged