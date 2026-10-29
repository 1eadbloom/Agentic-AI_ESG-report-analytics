"""
solution_agent.py — 策略生成代理（Solution Agent）

接收 DiagnosisResult，產出：
  1. 短/中/長期改善策略（LLM 生成，聚焦弱點）
  2. 同產業最佳實踐（RAG 從知識庫檢索 + LLM 整合）
  3. GRI/TCFD 對標缺口分析
"""
from __future__ import annotations
from dataclasses import dataclass, field
from interpretation_agent import DiagnosisResult, PillarAnalysis, AnalysisItem
from esg_utils import call_llm, GRI_STANDARDS, get_collection, get_embedding_fn
from citation import Citation, union_citations

_PILLAR_NAMES = {"E": "環境", "S": "社會", "G": "治理"}

# ── 資料結構 ──────────────────────────────────────────────────────
@dataclass
class StrategyItem:
    pillar:   str
    term:     str     # short / mid / long
    action:   str
    rationale: str = ""
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pillar": self.pillar, "term": self.term,
            "action": self.action, "rationale": self.rationale,
            "sources": [c.to_dict() if isinstance(c, Citation) else c
                        for c in self.citations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyItem":
        return cls(
            pillar=d.get("pillar", ""), term=d.get("term", "short"),
            action=d.get("action", ""), rationale=d.get("rationale", ""),
            citations=[
                c if isinstance(c, Citation) else Citation.from_dict(c)
                for c in d.get("sources", [])
            ],
        )


@dataclass
class SolutionResult:
    company:    str
    strategies: list[StrategyItem] = field(default_factory=list)
    best_practices: list[dict]     = field(default_factory=list)
    gri_gaps:       list[str]      = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "strategies": [s.to_dict() for s in self.strategies],
            "best_practices": self.best_practices,
            "gri_gaps": self.gri_gaps,
        }


# ── 策略生成（LLM）────────────────────────────────────────────────
import re

_TERM_LABELS = {
    "short": "短期（1 年內可執行）",
    "mid":   "中期（2–3 年）",
    "long":  "長期（3–5 年以上）",
}
_TERM_MAP = {"短期": "short", "中期": "mid", "長期": "long"}

# 每個面向的備援策略：僅在 LLM 兩次嘗試都解析不出任何策略時使用，並明確標示
_FALLBACK_STRATEGIES = {
    "E": [("short", "建立碳排放盤查制度，完成 GHG 盤查", "符合 GRI 305 要求"),
          ("mid",   "設定科學基礎減碳目標（SBTi）",       "對應 TCFD 策略揭露"),
          ("long",  "達成營運碳中和目標",                  "符合 2050 淨零承諾")],
    "S": [("short", "強化員工職安衛數據揭露完整性",         "補齊 GRI 403 缺漏"),
          ("mid",   "推動 DEI（多元共融）政策並量化目標",   "符合 GRI 405 精神")],
    "G": [("short", "補充獨立董事 ESG 相關訓練時數揭露",   "強化治理透明度"),
          ("mid",   "將 ESG KPI 納入高階主管績效考核",     "最佳實踐常見作法")],
}
_FALLBACK_NOTE = "（預設範本，非針對本公司分析）"

_GENERIC_LABELS = {"行動", "行動內容", "策略", "策略名稱", "具體行動"}

# 單行策略：[E][短期] 策略名稱：理由
_STRATEGY_LINE_RE = re.compile(
    r"\[([ESG])\]\s*\[([^\]]+)\]\s*(.+?)\s*(?:：|:)\s*(.+)")


def _weak_text(w) -> str:
    if isinstance(w, AnalysisItem):
        return w.text
    if isinstance(w, dict):
        return w.get("text", "")
    return str(w)


def _is_real_text(t: str) -> bool:
    """排除佔位字串與標題殘渣（###、---）。"""
    if len(re.sub(r"[#*\-_\s]", "", t)) < 6:
        return False
    return not t.startswith(("（無法評估", "（分析結果未包含", "（本份報告書"))


def _pillar_weakness_citations(pa: PillarAnalysis) -> list[Citation]:
    merged = union_citations([list(w.citations) for w in pa.weaknesses])
    real = [c for c in merged if not c.is_unattributed]
    return real or merged


def _parse_strategy_lines(raw: str, pillar: str, cites: list[Citation]) -> list[StrategyItem]:
    """容錯解析：去掉 Markdown 符號、項目符號與編號後再比對格式。"""
    out: list[StrategyItem] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        line = re.sub(r"[*_`]", "", line).strip()
        line = re.sub(r"^(?:[-•·]|\d+[.)、])\s*", "", line)
        m = _STRATEGY_LINE_RE.match(line)
        if not m or m.group(1) != pillar:       # 只收當次要求的面向
            continue
        _, term_zh, action, rationale = m.groups()
        term = next((v for k, v in _TERM_MAP.items() if k in term_zh), None)
        action = action.strip()
        # 容錯：模型照抄標籤「行動：策略：理由」時，以下一個冒號重新切分
        if action in _GENERIC_LABELS and re.search(r"[：:]", rationale):
            action, rationale = re.split(r"[：:]", rationale, maxsplit=1)
            action = action.strip()
        if term is None or action in seen:
            continue
        seen.add(action)
        out.append(StrategyItem(
            pillar=pillar, term=term, action=action,
            rationale=rationale.strip(),
            citations=list(cites),            # 繼承該面向弱點引用
        ))
    return out


def _generate_pillar_strategies(
    diagnosis: DiagnosisResult, pillar: str, cites: list[Citation],
    model: str | None,
) -> list[StrategyItem]:
    """單一面向各呼叫一次（5 項），輸出短、不易被 token 上限截斷。"""
    pa: PillarAnalysis = getattr(diagnosis, pillar)
    name = _PILLAR_NAMES[pillar]
    weak = [t for t in (_weak_text(w) for w in pa.weaknesses) if _is_real_text(t)]
    items = "\n".join(f"  - {t}" for t in weak) if weak else \
        "  - （診斷未找到具體待改善事項，請依 GRI/TCFD 通用最佳實務提出）"

    prompt = f"""你是「{diagnosis.company}」的企業永續策略顧問。
以下是「{name}（{pillar}）」面向的 ESG 診斷待改善事項：
{items}

請只針對「{name}（{pillar}）」面向提出 5 項具體可行的改善策略：
- 短期（1年內）：2 項
- 中期（2-3年）：2 項
- 長期（3-5年）：1 項（對應 GRI/TCFD/SBTi 等國際標準）

輸出規則：只輸出這 5 行，不要標題、說明、空行或 Markdown 符號。
每行格式為「[{pillar}][期程] 策略名稱：理由」：
- 期程只能是「短期」「中期」「長期」
- 冒號前面是簡短的策略名稱（不要寫「行動」兩個字），冒號後面才是一句話的理由
"""
    best: list[StrategyItem] = []
    for attempt in (1, 2):
        raw = call_llm(prompt, model=model, max_tokens=4096)
        parsed = _parse_strategy_lines(raw, pillar, cites)
        if not parsed:   # 除錯：看模型到底回了什麼（空字串＝可能被 token 上限截斷）
            print(f"    [debug] {pillar} 回應長度={len(raw or '')}，開頭：{(raw or '')[:200]!r}")
        if len(parsed) > len(best):
            best = parsed
        if len(best) >= 3:
            break
        print(f"    ⚠️ {pillar} 面向第 {attempt} 次只解析出 {len(parsed)} 條，重試 ...")

    if not best:   # 兩次都失敗：只對該面向使用預設範本，並明確標示
        print(f"    ⚠️ {pillar} 面向改用預設範本")
        best = [StrategyItem(pillar, term, action, rationale + _FALLBACK_NOTE,
                             list(cites))
                for term, action, rationale in _FALLBACK_STRATEGIES[pillar]]
    return best


def _is_fallback(st: StrategyItem) -> bool:
    return st.rationale.endswith(_FALLBACK_NOTE)


def generate_strategies(
    diagnosis: DiagnosisResult,
    model: str | None = None,
    existing: list[StrategyItem] | None = None,
) -> list[StrategyItem]:
    """existing：上一輪的策略。某面向已有 >= 3 條「非預設範本」策略就直接沿用，
    只重新生成缺漏的面向（QC 重試時省下大量時間）。"""
    pillar_cites = {p: _pillar_weakness_citations(getattr(diagnosis, p))
                    for p in ["E", "S", "G"]}
    strategies: list[StrategyItem] = []
    for p in ["E", "S", "G"]:
        kept = [st for st in (existing or [])
                if st.pillar == p and not _is_fallback(st)]
        if len(kept) >= 3:
            print(f"    {p} 面向：沿用上一輪的 {len(kept)} 條策略")
            strategies += kept
            continue
        got = _generate_pillar_strategies(diagnosis, p, pillar_cites[p], model)
        print(f"    {p} 面向：{len(got)} 條策略")
        strategies += got
    return strategies


# ── RAG 最佳實踐（從知識庫檢索）─────────────────────────────────
def get_best_practices(
    diagnosis: DiagnosisResult,
    model: str | None = None,
) -> list[dict]:
    ef = get_embedding_fn()
    _, collection = get_collection(ef)

    if collection.count() == 0:
        return _fallback_best_practices()

    practices = []
    for pillar in ["E", "S", "G"]:
        pa: PillarAnalysis = getattr(diagnosis, pillar)
        weaknesses_str = "、".join(_weak_text(w) for w in pa.weaknesses[:2]) if pa.weaknesses else ""
        query = f"{_PILLAR_NAMES[pillar]} ESG 最佳實踐 改善 {weaknesses_str}"

        # 知識庫最佳實踐查詢：只查 knowledge_base 文件，不混入公司報告
        try:
            res = collection.query(
                query_texts=[query], n_results=3,
                where={"doc_type": {"$eq": "knowledge_base"}},
            )
        except Exception:
            # 若知識庫尚未建立，fallback 到全庫查詢
            res = collection.query(query_texts=[query], n_results=3)
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []

        used_fallback = not docs
        if docs:
            context = "\n---\n".join(docs[:3])
            prompt = f"""根據以下從知識庫檢索到的 ESG 參考資料，
針對「{diagnosis.company}」在「{_PILLAR_NAMES[pillar]}」面向的改善需求（{weaknesses_str}），
提供 1-2 個具體的最佳實踐案例建議（50 字以內）。

參考資料：
{context[:1500]}

請直接輸出建議，不需前言。"""
            suggestion = (call_llm(prompt, model=model, max_tokens=1024) or "").strip()
            if not suggestion:      # 回應為空（例如思考模型把 token 用光）：不能留空白，改用預設範本並標示
                used_fallback = True
                suggestion = _fallback_best_practices()[
                    ["E", "S", "G"].index(pillar)
                ]["suggestion"]
        else:
            suggestion = _fallback_best_practices()[
                ["E", "S", "G"].index(pillar)
            ]["suggestion"]

        if used_fallback:
            print(f"    ⚠️ {pillar} 面向的最佳實踐改用預設範本（知識庫無資料或 LLM 無回應）")
        practices.append({
            "pillar": pillar,
            "suggestion": suggestion,
            "source": "預設範本（知識庫無資料或 LLM 無回應）" if used_fallback
                      else (metas[0].get("source_file", "知識庫") if metas else "知識庫"),
        })

    return practices


def _fallback_best_practices() -> list[dict]:
    return [
        {"pillar": "E", "suggestion": "參考同業領先企業作法：導入 ISO 50001 能源管理系統，承諾 RE100 並採購 PPA 再生能源，將綠電比例納入年度 KPI。", "source": "預設範本"},
        {"pillar": "S", "suggestion": "參考同業最佳實踐：建立員工心理健康支持方案（EAP），設定女性主管比例目標，並將職安事故率納入高階主管績效考核。", "source": "預設範本"},
        {"pillar": "G", "suggestion": "對標 TWSE 公司治理評鑑優良企業：將 ESG 風險納入整體風險管理架構，強化供應商 ESG 稽核機制，並每年由第三方驗證永續資料。", "source": "預設範本"},
    ]


# ── GRI/TCFD 對標缺口分析（規則式）──────────────────────────────
_GRI_TOPIC_KEYWORDS = {
    "GRI 302": ["能源", "用電", "耗電", "電力", "節電", "kWh", "度"],
    "GRI 303": ["用水", "水資源", "廢水", "取水"],
    "GRI 305": ["排放", "溫室氣體", "碳", "tCO2e"],
    "GRI 306": ["廢棄物", "回收"],
    "GRI 401": ["離職", "留任", "新進", "員工人數", "就業", "流動"],
    "GRI 403": ["職災", "職業災害", "職安", "工傷", "安全衛生"],
    "GRI 404": ["訓練", "培訓", "教育", "時數"],
    "GRI 413": ["社區", "公益", "在地", "志工", "捐"],
    "GRI 205": ["反貪", "貪腐", "賄賂", "誠信"],
    "GRI 206": ["反競爭", "公平競爭", "壟斷"],
    "GRI 418": ["個資", "隱私", "資安", "資訊安全"],
}


def analyze_gri_gaps(diagnosis: DiagnosisResult) -> list[str]:
    """啟發式：某 GRI 主題的關鍵字在該面向的 KPI（名稱＋數值）中完全沒出現，就列為可能缺口。
    注意這只代表「抽取到的 KPI 沒有」，不等於公司報告書沒有揭露。"""
    gaps = []
    for pillar in ["E", "S", "G"]:
        pa: PillarAnalysis = getattr(diagnosis, pillar)
        kpi_text = " ".join(f"{k.name}{k.value}{k.unit}" for k in pa.kpis)
        for std in GRI_STANDARDS[pillar]:
            code = " ".join(std.split(" ")[:2])          # 例："GRI 305"
            kws = _GRI_TOPIC_KEYWORDS.get(code, [])
            if kws and not any(k in kpi_text for k in kws):
                gaps.append(f"{std}：抽取到的 KPI 中未見相關量化指標，揭露可能不完整，建議補充")
    if not gaps:
        gaps = ["各 GRI 主題皆有對應的量化 KPI，建議進一步對標 TCFD 氣候風險情境分析"]
    return gaps


# ── 主流程 ────────────────────────────────────────────────────────
def run_solution_agent(
    diagnosis: DiagnosisResult,
    model: str | None = None,
    existing: list[StrategyItem] | None = None,
) -> SolutionResult:
    print("\n💡 Solution Agent 啟動")

    result = SolutionResult(company=diagnosis.company)

    print("  生成短/中/長期改善策略 ...")
    result.strategies = generate_strategies(diagnosis, model, existing)

    print("  從知識庫檢索同產業最佳實踐 ...")
    result.best_practices = get_best_practices(diagnosis, model)

    print("  進行 GRI/TCFD 對標缺口分析 ...")
    result.gri_gaps = analyze_gri_gaps(diagnosis)

    print(f"✅ Solution Agent 完成：{len(result.strategies)} 條策略建議\n")
    return result
