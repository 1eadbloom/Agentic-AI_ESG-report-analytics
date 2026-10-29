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


# ── 數值比對（讓引用指向「真正含有該數字」的片段）──────────────────
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_YEAR_RE = re.compile(r"(19|20)\d{2}")
NUMBER_BONUS = 0.15      # 片段含有結論中的數字時，排序加分


_CN_DIGIT = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}
# 只轉換「後面緊接單位」的中文數字（例：十日、七席、二十五人），
# 避免把「統一」「十分」這類一般詞彙誤轉成數字
_CN_NUM_RE = re.compile(
    r"(?<![一二兩三四五六七八九十百])"
    r"([一二兩三四五六七八九]?十[一二三四五六七八九]?|[一二兩三四五六七八九])"
    r"(?=\s*[日天人次席名位家件年月個場項])")


def _cn_to_int(s: str) -> int:
    if "十" in s:
        a, b = s.split("十")
        return (_CN_DIGIT.get(a, 1) if a else 1) * 10 + (_CN_DIGIT.get(b, 0) if b else 0)
    return _CN_DIGIT[s]


def _norm_for_numbers(t: str) -> str:
    """只移除千分位逗號、全形百分號轉半形，並把帶單位的中文數字轉成阿拉伯數字。

    注意不能把空白與換行一起刪掉：表格的儲存格常是一格一行（43,319／81,957／66,827），
    全部黏在一起會變成 433198195766827，反而比對不到任何數字。
    中文數字：原文寫「陪產假共十日」，KPI 寫成「10 天」，語意正確，不能因為字面不同就判成找不到。
    """
    t = (t or "").replace("％", "%")
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    return _CN_NUM_RE.sub(lambda m: str(_cn_to_int(m.group(1))), t)


def value_tokens(text: str) -> list[str]:
    """取出可用來比對的數字：至少 2 位數字，且排除年份（年份在報告中到處都是）。"""
    toks = []
    for m in _NUM_RE.findall(text or ""):
        n = m.replace(",", "").rstrip(".")
        if len(n.replace(".", "")) >= 2 and not _YEAR_RE.fullmatch(n):
            toks.append(n)
    return toks


def primary_token(value: str) -> str | None:
    """KPI 數值中用來比對的數字。優先取 2 位數以上；沒有時，若整個數值只有一個數字（例：「1」、「3 小時」）
    也可使用——否則個位數的 KPI 既無法驗證、也會被引用到不相干的頁面。"""
    toks = value_tokens(value)
    if toks:
        return toks[0]
    singles = [m.replace(",", "").rstrip(".") for m in _NUM_RE.findall(value or "")]
    singles = [n for n in singles if n and not _YEAR_RE.fullmatch(n)]
    return singles[0] if len(singles) == 1 else None


def contains_number(section_text: str, token: str) -> bool:
    """段落是否含有該數字（前後不得緊接其他數字，避免 38 比對到 238）。"""
    return re.search(rf"(?<![\d.]){re.escape(token)}(?!\d)",
                     _norm_for_numbers(section_text)) is not None


def _report_candidates(sections: list, run_company: str, run_year: str) -> list:
    return [s for s in sections
            if _in_scope(s, run_company, run_year)
            and getattr(s, "doc_type", "report") == "report"]


def value_found_in_sections(value: str, sections: list, run_company: str,
                            run_year: str = "") -> bool | None:
    """該數值是否出現在任一候選片段。沒有可比對的數字時回傳 None（無法判斷）。"""
    tok = primary_token(value)
    if tok is None:
        return None
    cands = _report_candidates(sections, run_company, run_year)
    return any(contains_number(getattr(s, "text", ""), tok) for s in cands)


# ── KPI 名稱與數值的語意鄰近檢查 ──────────────────────────────────
def name_trigrams(name: str) -> set[str]:
    """KPI 名稱中連續 3 個中文字的集合（括號內的補充說明不計）。"""
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", re.sub(r"[（(].*?[）)]", "", name or ""))
    return {cjk[i:i + 3] for i in range(len(cjk) - 2)}


def value_near_name_in_texts(name: str, value: str, texts: list[str],
                             win: int = 100) -> bool | None:
    """KPI 的數值是否出現在「名稱關鍵字附近」。

    只比對數字字串會漏掉「數字存在、但意思錯配」的 KPI（例：董事長室配置資源＝100%，
    其實 100% 是旁邊「完訓率」的數字）。判斷方式：
      - 句子裡的數值：前後 win 字內必須出現名稱的連續 3 個中文字
      - 表格裡的數值（單獨一行）：名稱與數值常相隔整個欄位，改為檢查整個片段是否含名稱關鍵字
    回傳 None 表示無法判斷（名稱不足 3 個中文字、數值沒有可比對的數字，
    或數值只出現在表格儲存格而名稱在相鄰 chunk）；回傳 False 才代表「有明確證據顯示不相鄰」。
    """
    grams = name_trigrams(name)
    tok = primary_token(value)
    if not grams or tok is None:
        return None
    pat = re.compile(rf"(?<![\d.]){re.escape(tok)}(?!\d)")
    saw_table_value = False
    saw_prose_without_name = False
    for text in texts:
        n = _norm_for_numbers(text)
        for m in pat.finditer(n):
            if any(g in n[max(0, m.start() - win):m.end() + win] for g in grams):
                return True
            ls = n.rfind("\n", 0, m.start()) + 1
            le = n.find("\n", m.end())
            le = len(n) if le < 0 else le
            alone = re.fullmatch(rf"\s*{re.escape(tok)}\s*[%‰]?\s*", n[ls:le]) is not None
            if alone:
                saw_table_value = True
                if any(g in n for g in grams):
                    return True
            else:
                saw_prose_without_name = True
    # 句子裡出現過這個數字、但旁邊沒有名稱 → 有明確證據顯示不相鄰
    if saw_prose_without_name:
        return False
    # 數值只出現在表格儲存格：欄位名稱可能在相鄰的 chunk，無法用鄰近度判斷，不下結論
    return None if saw_table_value else False


# ── 目標 vs 實績 ──────────────────────────────────────────────────
# KPI 區塊要放「已經發生的實績」。同一個數字在報告裡可能是承諾（「設定完訓率為100%」、
# 「長期（2035 年）累計減碳63%」），也可能是結果（「完成訓練，完訓率為100%」），兩者意思完全不同。
_TARGET_CUES = ("設定", "訂定", "目標", "預計", "預定", "規劃", "擬", "期程",
                "短期", "中期", "長期", "承諾", "將於")
_ACTUAL_CUES = ("完成", "達成", "已", "實際", "共計", "合計", "共有")


def context_is_target(text: str, pos: int, win: int = 30) -> bool:
    """數值前 win 字內有「設定、目標、長期…」之類的字眼，且沒有「完成、達成、合計…」的字眼。

    寧可漏判也不要誤判：誤判會把正確的實績 KPI 刪掉。例如「參考…規範制定《辦法》，2024 年立案之
    客訴案件共56 件」中的 56 是實績，所以「制定」不列入目標字眼，緊鄰數值的「共」視為實績字眼。"""
    left = text[max(0, pos - win):pos]
    if "共" in left[-4:]:
        return False
    return any(c in left for c in _TARGET_CUES) and not any(c in left for c in _ACTUAL_CUES)


def value_is_target_in_texts(name: str, value: str, texts: list[str],
                             win: int = 100) -> bool | None:
    """該 KPI 的數值，在「名稱附近」出現的每一處是否都是目標的語境。
    True＝都是目標（應捨棄）；False＝至少有一處不是目標；None＝無法判斷（名稱或數值不足以比對，或附近沒有出現）。"""
    grams = name_trigrams(name)
    tok = primary_token(value)
    if not grams or tok is None:
        return None
    pat = re.compile(rf"(?<![\d.]){re.escape(tok)}(?!\d)")
    verdicts = []
    for text in texts:
        n = _norm_for_numbers(text)
        for m in pat.finditer(n):
            if any(g in n[max(0, m.start() - win):m.end() + win] for g in grams):
                verdicts.append(context_is_target(n, m.start()))
    if not verdicts:
        return None
    return all(verdicts)


def kpi_is_target_by_citations(name: str, value: str, citations: list,
                               sections: list) -> bool | None:
    texts = texts_for_citations(citations, sections)
    return value_is_target_in_texts(name, value, texts) if texts else None


def texts_for_citations(citations: list, sections: list) -> list[str]:
    """依 chunk_id 取出被引用片段的全文（sections 可為物件或 dict）。"""
    def _get(s, k, default=""):
        return s.get(k, default) if isinstance(s, dict) else getattr(s, k, default)
    by_id = {str(_get(s, "section_id")): _get(s, "text") for s in sections}
    return [by_id[c.chunk_id] for c in citations
            if not c.is_unattributed and c.chunk_id in by_id]


def kpi_supported_by_citations(name: str, value: str, citations: list,
                               sections: list, win: int = 100) -> bool | None:
    texts = texts_for_citations(citations, sections)
    if not texts:
        return None
    return value_near_name_in_texts(name, value, texts, win)


# ── 來源選取（R1 + FR-005）────────────────────────────────────────
def select_sources(text: str, sections: list, run_company: str,
                   run_year: str = "", top_k: int = DEFAULT_TOP_K,
                   threshold: float = DEFAULT_THRESHOLD) -> list[Citation]:
    """將一則結論確定性地歸因給候選 section。

    - 只接受 run scope 內、doc_type == "report" 的片段（KB 不混入公司結論）
    - 以 similarity 排序；結論含有數字時，含該數字的片段加分（NUMBER_BONUS）
    - similarity >= threshold，或含有結論中的數字者才合格，取前 top_k 名
    - 沒有任何合格片段時標記 unattributed（FR-006）
    """
    candidates = _report_candidates(sections, run_company, run_year)
    if not candidates:
        return [make_unattributed(run_company, run_year)]

    toks = value_tokens(text)
    scored = score_against_sections(text, candidates)
    ranked = []
    for s, sc in scored:
        has_num = bool(toks) and any(contains_number(getattr(s, "text", ""), t)
                                     for t in toks)
        if sc >= threshold or has_num:
            ranked.append((sc + (NUMBER_BONUS if has_num else 0.0), s))
    ranked.sort(key=lambda x: x[0], reverse=True)
    picks = [s for _, s in ranked[:top_k]]
    if not picks:
        return [make_unattributed(run_company, run_year)]

    return [make_citation(s, excerpt=truncate(getattr(s, "text", "")))
            for s in picks]


def select_sources_for_value(text: str, value: str, sections: list,
                             run_company: str, run_year: str = "",
                             top_k: int = DEFAULT_TOP_K,
                             threshold: float = DEFAULT_THRESHOLD) -> list[Citation]:
    """KPI 專用：先找「文字真的含有該數值」的片段，再依與 KPI 描述的相似度排序。

    原本只用語意相似度，會把 KPI 引用到「主題相近、但沒有這個數字」的片段
    （例如把 526 人次引用到 p102，實際數字在 p63 的表格）。
    找不到含該數值的片段時，退回 select_sources（數值可能是 LLM 推算而非原文）。
    """
    tok = primary_token(value)
    if tok is not None:
        cands = _report_candidates(sections, run_company, run_year)
        containing = [s for s in cands
                      if contains_number(getattr(s, "text", ""), tok)]
        # 個位數（1、3、7…）幾乎每個片段都有，只看「含有該數字」沒有意義；
        # 改為要求名稱的關鍵字就在數字旁邊（40 字內，或表格儲存格且名稱在同一片段），
        # 否則會引用到只是剛好有個「第1次」的不相干頁面（真實案例：資料品質分數＝1 被引用到 p26-27，實際在 p58 的表格）
        if tok.isdigit() and len(tok) == 1 and containing:
            kpi_name = re.split(r"[：:]", text, maxsplit=1)[0]
            adjacent = [s for s in containing
                        if value_near_name_in_texts(kpi_name, value,
                                                    [getattr(s, "text", "")], win=40) is True]
            containing = adjacent or containing
        if containing:
            scored = score_against_sections(text, containing)
            return [make_citation(s, excerpt=truncate(getattr(s, "text", "")))
                    for s, _ in scored[:top_k]]
    return select_sources(text, sections, run_company, run_year, top_k, threshold)


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