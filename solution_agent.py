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
_TERM_LABELS = {
    "short": "短期（1 年內可執行）",
    "mid":   "中期（2–3 年）",
    "long":  "長期（3–5 年以上）",
}

def _weak_text(w) -> str:
    if isinstance(w, AnalysisItem):
        return w.text
    if isinstance(w, dict):
        return w.get("text", "")
    return str(w)


def _pillar_weakness_citations(pa: PillarAnalysis) -> list[Citation]:
    merged = union_citations([list(w.citations) for w in pa.weaknesses])
    real = [c for c in merged if not c.is_unattributed]
    return real or merged


def generate_strategies(
    diagnosis: DiagnosisResult,
    model: str | None = None,
) -> list[StrategyItem]:

    pillar_cites = {p: _pillar_weakness_citations(getattr(diagnosis, p))
                    for p in ["E", "S", "G"]}

    weaknesses_summary = ""
    for p in ["E", "S", "G"]:
        pa: PillarAnalysis = getattr(diagnosis, p)
        if pa.weaknesses:
            items = "\n".join(f"  - {_weak_text(w)}" for w in pa.weaknesses)
            weaknesses_summary += f"\n{_PILLAR_NAMES[p]}面向待改善事項：\n{items}\n"

    prompt = f"""你是「{diagnosis.company}」的企業永續策略顧問。
根據以下 ESG 診斷中發現的待改善事項，為企業提供具體可行的永續發展改善策略。

{weaknesses_summary}

請針對環境（E）、社會（S）、治理（G）三大面向，各提出：
- 短期（1年內）：2 項立即可執行的具體行動
- 中期（2-3年）：2 項需要系統建置的中期策略
- 長期（3-5年）：1 項對應國際標準（GRI/TCFD/SBTi）的長期目標

輸出格式（每行一項，格式為「[面向][期程] 行動：理由」）：
[E][短期] ...
[S][中期] ...
以此類推，共 15 項。
"""
    raw = call_llm(prompt, model=model)

    strategies = []
    import re
    for line in raw.splitlines():
        m = re.match(r"\[([ESG])\]\[([^]]+)\]\s*(.+?)(?:：|:)(.+)", line)
        if m:
            p, term_zh, action, rationale = m.groups()
            term_map = {"短期": "short", "中期": "mid", "長期": "long"}
            term = next((v for k, v in term_map.items() if k in term_zh), "short")
            strategies.append(StrategyItem(
                pillar=p, term=term,
                action=action.strip(),
                rationale=rationale.strip(),
                citations=list(pillar_cites.get(p, [])),  # 繼承該面向弱點引用
            ))

    # fallback：若解析失敗，提供預設策略（同樣帶上繼承引用）
    if not strategies:
        strategies = [
            StrategyItem("E", "short", "建立碳排放盤查制度，完成 GHG 盤查", "符合 GRI 305 要求", list(pillar_cites.get("E", []))),
            StrategyItem("E", "mid",   "設定科學基礎減碳目標（SBTi）",       "對應 TCFD 策略揭露", list(pillar_cites.get("E", []))),
            StrategyItem("E", "long",  "達成營運碳中和目標",                  "符合 2050 淨零承諾", list(pillar_cites.get("E", []))),
            StrategyItem("S", "short", "強化員工職安衛數據揭露完整性",         "補齊 GRI 403 缺漏", list(pillar_cites.get("S", []))),
            StrategyItem("S", "mid",   "推動 DEI（多元共融）政策並量化目標",   "符合 GRI 405 精神", list(pillar_cites.get("S", []))),
            StrategyItem("G", "short", "補充獨立董事 ESG 相關訓練時數揭露",   "強化治理透明度", list(pillar_cites.get("G", []))),
            StrategyItem("G", "mid",   "將 ESG KPI 納入高階主管績效考核",     "最佳實踐常見作法", list(pillar_cites.get("G", []))),
        ]
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

        if docs:
            context = "\n---\n".join(docs[:3])
            prompt = f"""根據以下從知識庫檢索到的 ESG 參考資料，
針對「{diagnosis.company}」在「{_PILLAR_NAMES[pillar]}」面向的改善需求（{weaknesses_str}），
提供 1-2 個具體的最佳實踐案例建議（50 字以內）。

參考資料：
{context[:1500]}

請直接輸出建議，不需前言。"""
            suggestion = call_llm(prompt, model=model, max_tokens=256)
        else:
            suggestion = _fallback_best_practices()[
                ["E", "S", "G"].index(pillar)
            ]["suggestion"]

        practices.append({
            "pillar": pillar,
            "suggestion": suggestion,
            "source": metas[0].get("source_file", "知識庫") if metas else "知識庫",
        })

    return practices


def _fallback_best_practices() -> list[dict]:
    return [
        {"pillar": "E", "suggestion": "參考台積電、中信金等領導企業作法：導入 ISO 50001 能源管理系統，承諾 RE100 並採購 PPA 再生能源，將綠電比例納入年度 KPI。", "source": "預設範本"},
        {"pillar": "S", "suggestion": "參考同業最佳實踐：建立員工心理健康支持方案（EAP），設定女性主管比例目標，並將職安事故率納入高階主管績效考核。", "source": "預設範本"},
        {"pillar": "G", "suggestion": "對標 TWSE 公司治理評鑑優良企業：將 ESG 風險納入整體風險管理架構，強化供應商 ESG 稽核機制，並每年由第三方驗證永續資料。", "source": "預設範本"},
    ]


# ── GRI/TCFD 對標缺口分析（規則式）──────────────────────────────
def analyze_gri_gaps(diagnosis: DiagnosisResult) -> list[str]:
    gaps = []
    for pillar in ["E", "S", "G"]:
        pa: PillarAnalysis = getattr(diagnosis, pillar)
        kpi_names = " ".join(k.name for k in pa.kpis).lower()
        for std in GRI_STANDARDS[pillar]:
            # 簡單判斷：若 KPI 中沒有相關關鍵字，視為可能缺漏
            std_key = std.split(" ")[1].lower() if " " in std else std.lower()
            if std_key not in kpi_names and len(pa.kpis) < 3:
                gaps.append(f"{std}：揭露資訊可能不完整，建議補充相關指標")
    if not gaps:
        gaps = ["整體揭露框架符合 GRI 基本要求，建議進一步對標 TCFD 氣候風險情境分析"]
    return gaps


# ── 主流程 ────────────────────────────────────────────────────────
def run_solution_agent(
    diagnosis: DiagnosisResult,
    model: str | None = None,
) -> SolutionResult:
    print("\n💡 Solution Agent 啟動")

    result = SolutionResult(company=diagnosis.company)

    print("  生成短/中/長期改善策略 ...")
    result.strategies = generate_strategies(diagnosis, model)

    print("  從知識庫檢索同產業最佳實踐 ...")
    result.best_practices = get_best_practices(diagnosis, model)

    print("  進行 GRI/TCFD 對標缺口分析 ...")
    result.gri_gaps = analyze_gri_gaps(diagnosis)

    print(f"✅ Solution Agent 完成：{len(result.strategies)} 條策略建議\n")
    return result
