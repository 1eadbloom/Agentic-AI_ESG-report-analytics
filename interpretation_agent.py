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
import os
import kpi_judge as _judge
from citation import (
    AnalysisItem, Citation, is_in_scope, make_citation, make_unattributed,
    score_against_sections, select_sources, truncate,
    select_sources_for_value, value_found_in_sections,
    kpi_supported_by_citations, texts_for_citations,
    kpi_is_target_by_citations, context_is_target,
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


# ── 解析輔助：清理 Markdown 殘渣、過濾無效 KPI ───────────────────
def _clean_llm_text(t: str) -> str:
    """清掉小模型常見的格式雜訊：LaTeX 包裝（$tCO_2e$）、重複量詞（人人次）。"""
    t = re.sub(r"\$([^$\n]{1,30})\$",
               lambda m: m.group(1).replace("_", "").replace("\\", ""), t)
    return t.replace("人人次", "人次")


def _strip_md(s: str) -> str:
    """移除 Markdown 強調符號與空白。"""
    return re.sub(r"[*_`#]", "", _clean_llm_text(s or "")).strip()


_BAD_NAME_START = ("及", "和", "並", "而", "或", "分別", "其中", "約", "達", "為", "共",
                   "另外", "此外", "同時", "因此", "此為")
_VALUE_START_RE = re.compile(r"^(約|逾|超過|達|近|>|≥|＞|~)?\s*\d")


_SPECULATION_RE = re.compile(r"[（(][^）)]*(可能|推測|需對照|疑似|應為|似乎|估計)[^）)]*[）)]")
_VAGUE_NAMES = {"調查結果", "結果", "數量", "數據", "比例", "比率", "人數", "件數", "金額", "情形", "狀況"}


def _clean_kpi_value(value: str) -> str:
    """去掉數值後面模型自己加的揣測（例：「1 分（滿分可能為較高分值，需對照內部標準）」→「1 分」）。
    這些話不是原文說的，放在 KPI 數值裡會被讀成報告的內容。"""
    return _SPECULATION_RE.sub("", value or "").strip()


def _starts_mid_phrase(text: str, m) -> bool:
    """名稱是否從「被換行切斷的詞中間」開始。
    例：「占總溫⏎室氣體排放量97%」會抓到「室氣體排放量」（少了「溫」），語意就變了。"""
    i = m.start("name")
    return i >= 2 and text[i - 1] == "\n" and "\u4e00" <= text[i - 2] <= "\u9fff"


def _clean_kpi_name(name: str) -> str:
    """去掉名稱尾端殘留的連接詞：「完訓率為」→「完訓率」、「總排放量約為」→「總排放量」。"""
    name = re.sub(r"\s*[（(]\d{1,3}[^\d）)年月日%][^）)]*[）)]", "", name or "")   # 例："(202決算)"
    return re.sub(r"(皆為|約為|為|達|約|共|計)$", "", name.strip()).strip()


def _valid_kpi(name: str, value: str) -> bool:
    """KPI 必須有名稱、數值以數字開頭，且名稱不是殘缺句
    （排除「• 進行「季檢視」及「: 360 度」、「分別為永康: 1 件」、「需參照…」這類結果）。"""
    name, value = (name or "").strip(), (value or "").strip()
    if len(name) < 2 or not re.search(r"[\u4e00-\u9fffA-Za-z]", name):
        return False
    if name[0] in "•·-–—*(（註" or name.startswith(_BAD_NAME_START):
        return False
    if name in _VAGUE_NAMES:                     # 例："調查結果＝8 件"（其實是「不受理的檢舉案件數」），讀者無法理解
        return False
    if re.search(r"大於|小於|高於|低於|超逾|超過|逾", name):   # 例："營收大於總營收"（政策排除門檻，不是績效數字）
        return False
    if name[-1].isdigit():                       # 例："金控1" + "00 %"（把 100% 從中間切開）
        return False
    if name.count("「") != name.count("」") or name.count("（") != name.count("）"):
        return False
    if name.count("(") != name.count(")"):               # 例："e) 及"
        return False
    if re.search(r"[：:➀-➉①-⑩]", name):                  # 例："2：相關產業定義：➀ 燃煤開採：營收超逾"（政策門檻，不是績效數字）
        return False
    return bool(_VALUE_START_RE.match(value))


_BULLET_RE = re.compile(r"^[-•·*]\s+(.*)")
_NOTE_RE = re.compile(r"^[*\s]*[（(]\s*註")


def _parse_bullets(block: str, min_len: int = 8) -> list[str]:
    """把 LLM 輸出的一個區塊解析成條目。

    - 跳過標題殘渣（###、---、***）與「（註：…）」備註
    - 縮排的子項目、換行延續的文字，併回上一個條目（避免「算」「驗」這種斷句碎片）
    - 太短的條目（< min_len 字）視為殘渣丟棄
    """
    lines = [ln for ln in block.splitlines() if ln.strip()]
    has_bullets = any(_BULLET_RE.match(ln) for ln in lines)
    items: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("【") or re.fullmatch(r"[#\-_=*\s]+", s):
            continue
        m = _BULLET_RE.match(ln)               # 只有「無縮排」的項目符號才算新條目
        body = _clean_llm_text((m.group(1) if m else re.sub(r"^[-•·*]\s+", "", s)).strip(" #"))
        if not body or _NOTE_RE.match(body):
            continue
        indented = ln.startswith((" ", "\t", "　"))
        prev_done = bool(items) and items[-1].rstrip().endswith(("。", "！", "？", "."))
        if m or not has_bullets or not items or (prev_done and not indented):
            items.append(body)   # 上一條已用句號收尾、這行又沒縮排 → 是新的條目，不是換行延續
        else:
            if len(body) >= 3:                 # 1-2 字的斷句碎片（如「算」「驗」）直接丟棄
                items[-1] += " " + body        # 縮排子項目或換行延續 → 併回上一項
    return [t for t in items if len(t) >= min_len]


# ── KPI 萃取（規則式，不耗用 LLM）────────────────────────────────
_KPI_RE = re.compile(
    r"(?P<name>[^\n，。！？,;；]{3,20}?)"
    r"(?P<value>\d[\d,]*(?:\.\d+)?)"   # 必須以數字開頭，避免把單獨的「，」「,」當成數值
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
            name  = _clean_kpi_name(_strip_md(m.group("name")).lstrip("，。、（("))
            value = m.group("value")
            unit  = m.group("unit") or ""
            if (len(name) >= 3 and unit and _valid_kpi(name, value)   # 規則式 KPI 必須有單位，排除 FTSE4Good、GRI 2021 這類名稱內的數字
                    and not _starts_mid_phrase(text, m)
                    and not context_is_target(text, m.start("value"))):
                kpis.append(KPIItem(name=name, value=value,
                                    unit=unit, pillar=pillar))
        return kpis[:10]

    by_name: dict[str, KPIItem] = {}
    for s in sections:
        s_text = _sec_text(s)
        for m in _KPI_RE.finditer(s_text):
            name  = _clean_kpi_name(_strip_md(m.group("name")).lstrip("，。、（("))
            value = m.group("value")
            unit  = m.group("unit") or ""
            if len(name) < 3 or not unit or not _valid_kpi(name, value):
                continue
            if _starts_mid_phrase(s_text, m) or context_is_target(s_text, m.start("value")):
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
_PILLAR_QUERY = {
    "E": "溫室氣體排放 碳排放 能源使用 水資源 廢棄物 再生能源 氣候風險 TCFD",
    "S": "員工訓練 職業安全衛生 多元共融 人權 供應鏈 社區參與 客戶權益",
    "G": "董事會 獨立董事 誠信治理 風險管理 反腐敗 股東權益 稽核",
}

def _numeric_density(text: str) -> int:
    """片段中的數字個數；表格與統計數據的片段會比一般敘述高很多。"""
    return len(re.findall(r"\d[\d,.]*", text or ""))


def _kpi_only_prompt(pillar: str, company: str, report_year: str, blob: str) -> str:
    return f"""你是一位專業的 ESG 永續顧問，正在閱讀「{company}」{report_year or '2024'} 年永續報告書中「{_PILLAR_NAMES[pillar]}」面向的段落節錄。

{blob}

請只列出文中「明確寫出數值」的關鍵績效指標，每行一項，格式為「指標名稱：數值 單位」。
規則：
- 指標名稱要完整，且數值必須與該名稱在原文中直接對應（同一句話，或同一個表格的對應列與欄）
- 只列已發生的實績；目標、計畫、申請條件、排除門檻不要列
- 沒有把握的不要列；最多 8 項；不要任何標題或說明
"""


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
    if len(sections) > 40:
        query = _PILLAR_QUERY.get(pillar, "")
        scored = score_against_sections(query, sections)
        sections = [s for s, _ in scored[:40]]

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
    if "\ufffd" in raw:     # 小模型偶爾會插入亂碼字元（原文本身沒有）；這是取樣雜訊，重問一次通常就消失
        print(f"    ⚠️ {_PILLAR_NAMES[pillar]}面向的回應含亂碼字元，重新生成一次")
        raw2 = call_llm(prompt, model=model)
        if raw2.count("\ufffd") <= raw.count("\ufffd"):
            raw = raw2

    def parse_section(tag: str, min_len: int = 8) -> list[str]:
        m = re.search(rf"【{tag}[^】]*】(.*?)(?=【|\Z)", raw, re.DOTALL)
        return _parse_bullets(m.group(1), min_len) if m else []

    highlights_raw = parse_section("亮點") or ["（分析結果未包含亮點資訊）"]
    weaknesses_raw = parse_section("待改善") or ["（分析結果未包含改善建議）"]
    # 模型的 KPI 區塊標題不一定照 prompt 寫（可能是「關鍵指標」「量化指標」「KPI」…），逐一嘗試
    kpis_raw: list[str] = []
    for _tag in ("關鍵績效指標", "關鍵指標", "量化指標", "績效指標", "KPI"):
        kpis_raw = parse_section(_tag, min_len=4)
        if kpis_raw:
            break

    # 模型有時完全不輸出 KPI 區塊（或只寫一兩行）。主要分析使用的是依相關度排序的前 3000 字，
    # 可能剛好沒包含數據表格；這時改用「數值密度高」的片段，單獨再問一次，只要求 KPI。
    if len(kpis_raw) < 3:
        why = "沒有 KPI 區塊" if not kpis_raw else f"只有 {len(kpis_raw)} 行"
        headers = re.findall(r"【([^】]{1,24})】", raw)
        print(f"    ⚠️ KPI（{pillar}）{why}（回應中的區塊標題：{headers}），改用數值密度高的片段單獨再問一次")
        if os.environ.get("ESG_DEBUG_KPI") == "1":
            print(f"      （原始回應結尾）{raw[-300:]!r}")
        try:
            ranked = sorted(sections, key=lambda x: _numeric_density(_sec_text(x)), reverse=True)
            blob2 = "\n\n".join(_sec_text(x) for x in ranked)[:4500]
            raw2 = call_llm(_kpi_only_prompt(pillar, company, report_year, blob2),
                            model=model, max_tokens=1024)
            kpis_raw = kpis_raw + [x for x in _parse_bullets(raw2, 4) if x not in kpis_raw]
        except Exception as e:           # KPI 補問失敗不應讓整個面向失敗
            print(f"      ⚠️ KPI 補問失敗：{type(e).__name__}: {str(e)[:120]}")
    summary_m  = re.search(r"【面向總結】(.*?)(?=【|\Z)", raw, re.DOTALL)
    summary    = summary_m.group(1).strip() if summary_m else ""

    highlights = [AnalysisItem(t, select_sources(t, sections, company, report_year))
                  for t in highlights_raw]
    weaknesses = [AnalysisItem(t, select_sources(t, sections, company, report_year))
                  for t in weaknesses_raw
                  if not t.startswith("（分析結果未包含")]

    # LLM 提供的 KPI：以相似度歸因
    kpis = []
    dropped: list[tuple[str, str, str]] = []     # (名稱, 數值, 捨棄原因)；讓每個決定都看得見
    for kline in kpis_raw:
        if not ("：" in kline or ":" in kline):
            dropped.append((kline[:24], "", "沒有冒號，無法拆成名稱與數值"))
            continue
        if True:
            parts = re.split(r"[：:]", kline, maxsplit=1)
            k_name = _clean_kpi_name(_strip_md(parts[0]))
            k_value = _clean_kpi_value(_strip_md(parts[1]))
            if not _valid_kpi(k_name, k_value):
                dropped.append((k_name, k_value, "名稱或數值不合格"))
                continue                      # 沒有數字的「KPI」不算（例如「[文本片段數值不完整]」）
            # 數值必須真的出現在被引用的片段；原文找不到的數字（例如 LLM 自己算的 3/7≈42.86%）要標示
            found = value_found_in_sections(k_value, sections, company, report_year)
            if found is False:
                k_value += "（推算，原文未直接出現）"
            cits = select_sources_for_value(
                kline, k_value, sections, company, report_year)
            if found is not False:      # 已標示「推算」的值本來就不在原文，不做鄰近檢查
                # 數字必須出現在「名稱關鍵字附近」，否則是模型把別的項目的數字錯配過來
                # （例：董事長室配置資源＝100%，其實 100% 是旁邊完訓率的數字）
                near = kpi_supported_by_citations(k_name, k_value, cits, sections)
                if near is False:
                    dropped.append((k_name, k_value, "名稱與數值不相鄰"))
                    continue
                # 選用：LLM 檢查員。實測它與確定性檢查在多數案例答案一致，
                # 只有「確定性檢查無法判斷」（表格儲存格、個位數）才有補位價值，所以只在那時才呼叫，
                # 一次執行通常只多 2 到 4 次呼叫，而不是每個 KPI 都問。
                if kpi_is_target_by_citations(k_name, k_value, cits, sections) is True:
                    dropped.append((k_name, k_value, "目標而非實績"))
                    continue
                if near is None and _judge.enabled():
                    ev = texts_for_citations(cits, sections)
                    if ev and _judge.judge_kpi(k_name, k_value, ev) is False:
                        dropped.append((k_name, k_value, "檢查員判定原文未明確支持"))
                        continue
            kpis.append(KPIItem(
                name=k_name, value=k_value, unit="", pillar=pillar,
                citations=cits))

    # 記錄 KPI 的處理結果：終端機（有被捨棄時，或 ESG_DEBUG_KPI=1）與 Langfuse trace
    if dropped or os.environ.get("ESG_DEBUG_KPI") == "1":
        print(f"    KPI（{pillar}）：模型輸出 {len(kpis_raw)} 行，保留 {len(kpis)}，捨棄 {len(dropped)}")
        for n, v, why in dropped:
            print(f"      ⚠️ 捨棄「{n}」＝{v[:16]}｜{why}")
    try:
        from tracing import get_client
        get_client().update_current_span(metadata={
            f"kpi_{pillar}_llm_lines": str(len(kpis_raw)),
            f"kpi_{pillar}_kept": str(len(kpis)),
            f"kpi_{pillar}_dropped": "; ".join(f"{n[:12]}={v[:8]}({why})" for n, v, why in dropped)[:400] or "none",
        })
    except Exception:
        pass

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
            # 查詢語句與「新 sections 路徑」一致，使用含主題詞的 _PILLAR_QUERY；
            # 原本只用「社會 公司名 永續報告」這種泛用查詢，會撈到無關片段（例如 S 面向全部引用同一頁）
            query = _PILLAR_QUERY.get(pillar) or f"{_PILLAR_NAMES[pillar]} {company} 永續報告"
            where = [{"doc_type": {"$eq": "report"}},
                     {"pillar": {"$eq": pillar}},
                     {"company": {"$eq": company}}]
            if report_year:
                where.append({"report_year": {"$eq": report_year}})
            if source_file:
                where.append({"source_file": {"$eq": source_file}})
            res = collection.query(
                query_texts=[query], n_results=40,   # 與 analyze_pillar 的 top-40 一致
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

        analysis = analyze_pillar(pillar, pillar_secs, company,
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