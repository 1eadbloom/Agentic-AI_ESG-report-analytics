"""
interpretation_agent.py — 內容理解與摘要代理（Interpretation Agent）

接收 Search Agent 的 sections，對每個 E/S/G 面向：
  1. 彙整該面向所有段落
  2. LLM 生成結構化摘要（亮點、弱點、KPI）
  3. 將每個結論確定性歸因到來源片段（見 citation.py / research R1/R3）
  4. 回傳 DiagnosisResult 供 Solution Agent 使用
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
from typing import Optional

from esg_utils import call_llm, GRI_STANDARDS, get_collection, get_embedding_fn
from citation import (
    AnalysisItem, Citation, is_in_scope, make_citation, make_unattributed,
    select_sources, truncate,
)

# ── 資料結構 ──────────────────────────────────────────────────────
@dataclass
class KPIItem:
    name:   str
    value:  str
    unit:   Optional[str] = None
    pillar: Optional[str] = None
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "value": self.value,
            "unit": self.unit, "pillar": self.pillar,
            "sources": [c.to_dict() if isinstance(c, Citation) else c
                        for c in self.citations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KPIItem":
        return cls(
            name=d.get("name", ""), value=d.get("value", ""),
            unit=d.get("unit"), pillar=d.get("pillar"),
            citations=[
                c if isinstance(c, Citation) else Citation.from_dict(c)
                for c in d.get("sources", [])
            ],
        )


@dataclass
class PillarAnalysis:
    pillar:     str          # E / S / G
    highlights: list[AnalysisItem] = field(default_factory=list)
    weaknesses: list[AnalysisItem] = field(default_factory=list)
    kpis:       list[KPIItem] = field(default_factory=list)
    raw_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "pillar": self.pillar,
            "highlights": [h.to_dict() for h in self.highlights],
            "weaknesses": [w.to_dict() for w in self.weaknesses],
            "kpis": [k.to_dict() for k in self.kpis],
            "raw_summary": self.raw_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PillarAnalysis":
        return cls(
            pillar=d.get("pillar", ""),
            highlights=[AnalysisItem.from_dict(x) if isinstance(x, dict) else x
                        for x in d.get("highlights", [])],
            weaknesses=[AnalysisItem.from_dict(x) if isinstance(x, dict) else x
                        for x in d.get("weaknesses", [])],
            kpis=[KPIItem.from_dict(x) if isinstance(x, dict) else x
                  for x in d.get("kpis", [])],
            raw_summary=d.get("raw_summary", ""),
        )


@dataclass
class DiagnosisResult:
    company:    str
    source_file: str
    E: PillarAnalysis = field(default_factory=lambda: PillarAnalysis("E"))
    S: PillarAnalysis = field(default_factory=lambda: PillarAnalysis("S"))
    G: PillarAnalysis = field(default_factory=lambda: PillarAnalysis("G"))
    overall_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "source_file": self.source_file,
            "E": self.E.to_dict(),
            "S": self.S.to_dict(),
            "G": self.G.to_dict(),
            "overall_summary": self.overall_summary,
        }


# ── section 讀取輔助 ─────────────────────────────────────────────
def _sec_text(s: object) -> str:
    return getattr(s, "text", "") if not isinstance(s, dict) else s.get("text", "")


def _sec_meta(s: object) -> tuple[str, str, str, str, str]:
    if isinstance(s, dict):
        return (str(s.get("company", "")), str(s.get("report_year", "")),
                str(s.get("source_file", "")), str(s.get("section_id", "")),
                str(s.get("page_range", "")), str(s.get("doc_type", "report")))
    return (str(getattr(s, "company", "")), str(getattr(s, "report_year", "")),
            str(getattr(s, "source_file", "")), str(getattr(s, "section_id", "")),
            str(getattr(s, "page_range", "")), str(getattr(s, "doc_type", "report")))


# ── KPI 萃取（規則式，不耗用 LLM）────────────────────────────────
_KPI_RE = re.compile(
    r"(?P<name>[^\n，。！？]{3,20}?)"
    r"(?P<value>[\d,，.]+(?:\.\d+)?)"
    r"\s*(?P<unit>%|噸|公噸|度|萬度|件|人|萬人|千人|元|萬元|億元|kWh|tCO2e|m³)?",
    re.UNICODE,
)

def extract_kpis(text: str, pillar: str,
                 sections: list | None = None) -> list[KPIItem]:
    """規則式 KPI 萃取。

    傳入 sections 時，直接對每個 section 的文字跑 regex，取得「精確溯源」
    （R1）：每個 KPI 的 citation 就是實際匹配到的 section。
    未傳 sections 時維持舊行為（無 citation）。
    """
    kpis: list[KPIItem] = []
    if not sections:
        for m in _KPI_RE.finditer(text):
            name  = m.group("name").strip().lstrip("，。、（(")
            value = m.group("value")
            unit  = m.group("unit") or ""
            if len(name) >= 3 and len(value) >= 1:
                kpis.append(KPIItem(name=name, value=value,
                                    unit=unit, pillar=pillar))
        return kpis[:10]

    by_name: dict[str, KPIItem] = {}
    for s in sections:
        s_text = _sec_text(s)
        for m in _KPI_RE.finditer(s_text):
            name  = m.group("name").strip().lstrip("，。、（(")
            value = m.group("value")
            unit  = m.group("unit") or ""
            if len(name) < 3 or len(value) < 1:
                continue
            sec = s if isinstance(s, dict) else s
            cit = make_citation(sec, excerpt=truncate(s_text))
            if name in by_name:
                item = by_name[name]
                if all(c.chunk_id != cit.chunk_id for c in item.citations):
                    item.citations.append(cit)
            else:
                by_name[name] = KPIItem(name=name, value=value, unit=unit,
                                        pillar=pillar, citations=[cit])
    return list(by_name.values())[:10]


# ── LLM 分析（每個面向各一次）────────────────────────────────────
_PILLAR_NAMES = {"E": "環境（Environmental）", "S": "社會（Social）", "G": "治理（Governance）"}
_GRI_MAP = {"E": "GRI 302/303/305/306", "S": "GRI 401/403/404/413", "G": "GRI 205/206/418"}

def analyze_pillar(pillar: str, sections: list,
                   company: str, report_year: str = "",
                   model: str | None = None) -> PillarAnalysis:
    """分析單一面向。sections 為該面向的候選片段（含 metadata）。

    亮點/弱點/KPI 皆以確定性方式歸因（R1）：亮點與弱點用 embedding 相似度
    top-k，KPI 用 regex 精確溯源。無法溯源者標記 unattributed（FR-006）。
    """
    sections = [s for s in (sections or []) if s is not None]
    # 只保留 run scope 內、doc_type == "report" 的片段（FR-004/SC-002，
    # KB 僅供 best-practices 使用，不進入公司結論）
    sections = [s for s in sections
                if is_in_scope(s, company, report_year)
                and getattr(s, "doc_type", "report") == "report"]
    text_blob = "\n\n".join(_sec_text(s) for s in sections)

    if not text_blob.strip():
        return PillarAnalysis(
            pillar=pillar,
            highlights=[AnalysisItem(
                "（本份報告書中未找到相關內容）",
                [make_unattributed(company, report_year)])],
            weaknesses=[AnalysisItem(
                "（無法評估）",
                [make_unattributed(company, report_year)])],
            raw_summary="",
        )

    prompt = f"""你是一位專業的 ESG 永續顧問，正在分析「{company}」的 {report_year or '2024'} 年永續報告書。

以下是該報告書中屬於「{_PILLAR_NAMES[pillar]}」面向的相關段落節錄：

{text_blob[:3000]}

請依照以下格式，輸出該面向的分析結果（使用繁體中文）：

【亮點 Highlights】（條列 3-5 點，企業已達成或明確承諾的正向作為）
- 
- 

【待改善 Weaknesses】（條列 2-4 點，揭露不足、目標模糊、或相較 {_GRI_MAP[pillar]} 標準缺漏之處）
- 
- 

【關鍵績效指標 KPIs】（條列已揭露的量化數據，格式：指標名稱：數值 單位）
- 
- 

【面向總結】（2-3 句話，整體評估）
"""
    raw = call_llm(prompt, model=model)

    def parse_section(tag: str) -> list[str]:
        m = re.search(rf"【{tag}[^】]*】(.*?)(?=【|\Z)", raw, re.DOTALL)
        if not m:
            return []
        return [ln.lstrip("- •·　").strip()
                for ln in m.group(1).strip().splitlines()
                if ln.strip() and not ln.strip().startswith("【")]

    highlights_raw = parse_section("亮點") or ["（分析結果未包含亮點資訊）"]
    weaknesses_raw = parse_section("待改善") or ["（分析結果未包含改善建議）"]
    kpis_raw   = parse_section("關鍵績效指標")
    summary_m  = re.search(r"【面向總結】(.*?)(?=【|\Z)", raw, re.DOTALL)
    summary    = summary_m.group(1).strip() if summary_m else ""

    highlights = [AnalysisItem(t, select_sources(t, sections, company, report_year))
                  for t in highlights_raw]
    weaknesses = [AnalysisItem(t, select_sources(t, sections, company, report_year))
                  for t in weaknesses_raw]

    # LLM 提供的 KPI：以相似度歸因
    kpis = []
    for kline in kpis_raw:
        if "：" in kline or ":" in kline:
            parts = re.split(r"[：:]", kline, maxsplit=1)
            kpis.append(KPIItem(
                name=parts[0].strip(), value=parts[1].strip(), pillar=pillar,
                citations=select_sources(kline, sections, company, report_year)))

    # 規則式補充 KPI：精確溯源
    kpis += extract_kpis(text_blob, pillar, sections=sections)
    # 去重（同名合併 citations）
    seen = {}
    for k in kpis:
        key = k.name.strip().lower()
        if key in seen:
            item = seen[key]
            for c in k.citations:
                if all(c.chunk_id != x.chunk_id for x in item.citations):
                    item.citations.append(c)
        else:
            seen[key] = KPIItem(name=k.name, value=k.value, unit=k.unit,
                                pillar=k.pillar, citations=list(k.citations))

    return PillarAnalysis(
        pillar=pillar,
        highlights=highlights,
        weaknesses=weaknesses,
        kpis=list(seen.values())[:8],
        raw_summary=summary,
    )


# ── 主流程 ────────────────────────────────────────────────────────
def _meta_section(doc: str, meta: dict, section_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        section_id=section_id or meta.get("section_id", ""),
        title=meta.get("title", ""),
        text=doc,
        pillar=meta.get("pillar", ""),
        source_file=meta.get("source_file", ""),
        page_range=meta.get("page_range", ""),
        company=meta.get("company", ""),
        report_year=meta.get("report_year", ""),
        doc_type=meta.get("doc_type", "report"),
    )


def _run_report_year(company: str, source_file: str, sections: list | None) -> str:
    if sections:
        for s in sections:
            c, y, *_ = _sec_meta(s)
            if c == company and y:
                return y
    m = re.search(r"(20[0-9]{2})", source_file or "")
    return m.group(1) if m else ""


def run_interpretation_agent(
    company: str,
    source_file: str,
    sections: list | None = None,   # ESGSection list（可選）
    model: str | None = None,
) -> DiagnosisResult:
    """
    若傳入 sections，直接用（並以 run scope 過濾）；否則從 ChromaDB 檢索
    （檢索過濾已強化：source_file + company + report_year + doc_type，見 T025）。
    """
    print("\n🔬 Interpretation Agent 啟動")

    result = DiagnosisResult(company=company, source_file=source_file)
    report_year = _run_report_year(company, source_file, sections)

    for pillar in ["E", "S", "G"]:
        print(f"  分析面向：{_PILLAR_NAMES[pillar]} ...")

        if sections:
            # 從傳入 sections 過濾面向（scope 由 citation.py 的守門把關）
            pillar_secs = [s for s in sections
                           if (hasattr(s, "pillar") and s.pillar == pillar)
                           or (isinstance(s, dict) and s.get("pillar") == pillar)]
        else:
            # 從 ChromaDB 檢索（強化隔離過濾：公司 + 年度 + 檔案 + 類型）
            ef = get_embedding_fn()
            _, collection = get_collection(ef)
            query = f"{_PILLAR_NAMES[pillar]} {company} 永續報告"
            where = [{"doc_type": {"$eq": "report"}},
                     {"company": {"$eq": company}}]
            if report_year:
                where.append({"report_year": {"$eq": report_year}})
            if source_file:
                where.append({"source_file": {"$eq": source_file}})
            res = collection.query(
                query_texts=[query], n_results=12,
                where={"$and": where} if len(where) > 1 else where[0],
                include=["documents", "metadatas"],
            )
            # section_id 存於 ChromaDB `id`（search_agent upsert ids），
            # 不在 metadata 中；必須由 ids 帶回，否則 citation chunk_id 為空（T033）
            ids   = res["ids"][0] if res.get("ids") else []
            docs  = res["documents"][0] if res["documents"] else []
            metas = res["metadatas"][0] if res["metadatas"] else []
            pillar_secs = [_meta_section(d, m, sid)
                           for d, m, sid in zip(docs, metas, ids)]

        analysis = analyze_pillar(pillar, pillar_secs[:15], company,
                                  report_year, model)
        setattr(result, pillar, analysis)

    # 整體摘要
    result.overall_summary = (
        f"【環境】{result.E.raw_summary} "
        f"【社會】{result.S.raw_summary} "
        f"【治理】{result.G.raw_summary}"
    ).strip()

    print("✅ Interpretation Agent 完成\n")
    return result