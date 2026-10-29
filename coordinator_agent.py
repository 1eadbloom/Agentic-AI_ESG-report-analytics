"""
coordinator_agent.py — 任務協調與品質控管代理（Coordinator Agent）

用 LangGraph 將四個 Agent 串接成有狀態的工作流程：
  START → search → interpret → solution → quality_check → report → END
                                              ↑ 若 QC 失敗 (最多 2 次重試)
                                              └──────────────────┘

用法：
  python coordinator_agent.py --file "2024年永續報告書(中)中信證券000616.pdf" --company "中信證券"
  python coordinator_agent.py --file xxx.pdf --rebuild   # 重建索引
  python coordinator_agent.py --file xxx.pdf --model ollama/gemma4-plan:latest

可觀測性（Langfuse tracing）：
  設定 .env 中的 LANGFUSE_* 後自動啟用；未設定時所有追蹤函式為無作用，不影響執行與離線測試。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

import esg_utils
from esg_utils import REPORTS_DIR, OUTPUTS_DIR
from search_agent        import run_search_agent, ESGSection
from interpretation_agent import run_interpretation_agent, DiagnosisResult
from solution_agent       import run_solution_agent, SolutionResult, StrategyItem
from citation            import citations_to_markdown

from tracing import observe, get_client, propagate_attributes


# ── 引用渲染輔助 ──────────────────────────────────────────────────
def _item_line(item) -> str:
    """Markdown 單行：純文字直接輸出；AnalysisItem（dict）補上內嵌引用。"""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        sources = item.get("sources", []) or []
    else:
        sources = getattr(item, "citations", []) or []
    text = item["text"] if isinstance(item, dict) else item.text
    return text + citations_to_markdown(sources)


# ── QC 輔助：判斷「佔位字串」與「上游失敗」 ───────────────────────
# 這些字串代表「沒有真正的分析內容」，不能被當成有內容
PLACEHOLDER_PREFIXES = (
    "（無法評估）",
    "（本份報告書中未找到相關內容）",
    "分析結果未包含改善建議",
    "（無資料）",
)


def _real_items(items) -> list:
    """濾掉佔位字串與標題殘渣（###、---），只留真正的分析內容。"""
    out = []
    for it in items or []:
        text = it.get("text", "") if isinstance(it, dict) else str(it)
        text = text.strip().lstrip("#*- ").strip()
        if len(text) >= 6 and not text.startswith(PLACEHOLDER_PREFIXES):
            out.append(it)
    return out


def _search_failed(state) -> bool:
    """Search 階段失敗（找不到檔案或例外）時，重跑下游沒有意義。"""
    return any(("找不到檔案" in e) or ("Search Agent 失敗" in e)
               for e in state.get("errors", []))


# ── Graph State ────────────────────────────────────────────────────
class ESGState(TypedDict):
    # 輸入
    file_name:   str
    company:     str
    model:       str
    rebuild:     bool
    # 中間結果
    sections:    list
    diagnosis:   dict
    solution:    dict
    # 控制
    retry_count: int
    qc_passed:   bool
    retry_target: str          # QC 失敗時要回到哪個節點："interpret" 或 "solution"
    errors:      list[str]
    # 最終輸出
    report_path: str


# ── Node 1：Search ─────────────────────────────────────────────────
@observe(capture_input=False, capture_output=False)
def node_search(state: ESGState) -> dict:
    print("=" * 60)
    print(f"[1/4] Search Agent — 解析報告：{state['file_name']}")
    try:
        p = REPORTS_DIR / state["file_name"]
        if not p.exists():
            available = sorted(x.name for x in REPORTS_DIR.glob("*.pdf"))
            hint = f"（data/reports/ 內的 PDF：{available}）" if available else ""
            return {"errors": state["errors"] + [f"找不到檔案：{p}{hint}"],
                    "sections": []}
        secs = run_search_agent(files=[p], rebuild=state["rebuild"],
                                   company=state["company"], include_kb=True)
        # 知識庫文件只供最佳實踐檢索使用，不能混進公司報告的分析範圍
        secs = [x for x in secs if getattr(x, "doc_type", "report") == "report"]
        get_client().update_current_span(
            metadata={"sections": str(len(secs)), "file": state["file_name"]}
        )
        return {"sections": [s.__dict__ if hasattr(s, "__dict__") else s
                              for s in secs],
                "errors": state["errors"]}
    except Exception as e:
        return {"errors": state["errors"] + [f"Search Agent 失敗：{e}"],
                "sections": []}


# ── Node 2：Interpret ──────────────────────────────────────────────
@observe(capture_input=False)
def node_interpret(state: ESGState) -> dict:
    print(f"[2/4] Interpretation Agent — 分析：{state['company']}")
    try:
        from search_agent import ESGSection as ES
        sections = [
            ES(**s) if isinstance(s, dict) else s
            for s in (state["sections"] or [])
        ]
        diagnosis = run_interpretation_agent(
            company=state["company"],
            source_file=state["file_name"],
            sections=sections if sections else None,
            model=state["model"],
        )
        return {"diagnosis": diagnosis.to_dict(), "errors": state["errors"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"errors": state["errors"] + [f"Interpretation Agent 失敗：{e}"],
                "diagnosis": {}}


# ── Node 3：Solution ───────────────────────────────────────────────
@observe(capture_input=False)
def node_solution(state: ESGState) -> dict:
    print(f"[3/4] Solution Agent — 生成改善策略")
    if not state.get("diagnosis"):
        # 上游（Interpretation）整個失敗時，用空診斷產生策略只是浪費時間，也會產出無依據的建議；
        # 直接略過，交給 QC 判定並回到 interpret 重試
        print("  ⚠️  沒有診斷結果（Interpretation 失敗），略過策略生成")
        return {"solution": {}, "errors": state["errors"]}
    try:
        from interpretation_agent import DiagnosisResult, PillarAnalysis
        d = state["diagnosis"]

        def _build_pa(pillar: str) -> PillarAnalysis:
            pd = d.get(pillar, {})
            if isinstance(pd, dict):
                pa = PillarAnalysis.from_dict(pd)
                pa.pillar = pillar
                return pa
            return PillarAnalysis(pillar=pillar)

        dr = DiagnosisResult(
            company=d.get("company", state["company"]),
            source_file=d.get("source_file", state["file_name"]),
            E=_build_pa("E"), S=_build_pa("S"), G=_build_pa("G"),
            overall_summary=d.get("overall_summary", ""),
        )
        # 只有「策略不足」觸發的重試，才沿用上一輪已成功的面向（診斷沒變）
        existing = None
        if state.get("retry_target") == "solution" and state["retry_count"] > 0:
            existing = [StrategyItem.from_dict(x) if isinstance(x, dict) else x
                        for x in state.get("solution", {}).get("strategies", [])]
        sol = run_solution_agent(dr, model=state["model"], existing=existing)
        return {"solution": sol.to_dict(), "errors": state["errors"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"errors": state["errors"] + [f"Solution Agent 失敗：{e}"],
                "solution": {}}


# ── Node 4：Quality Check ──────────────────────────────────────────
@observe()
def node_quality_check(state: ESGState) -> dict:
    print(f"[QC]  品質驗證（第 {state['retry_count']+1} 次）")
    issues = []
    d = state.get("diagnosis", {})
    s = state.get("solution", {})

    # 上游失敗：沒有任何可分析的內容，重試無效
    if _search_failed(state):
        issues.append("Search Agent 失敗，沒有可分析的內容（重試無效）")

    # 檢查 E/S/G 三面向都有「真正的」分析結果（佔位字串不算）
    for pillar in ["E", "S", "G"]:
        pd = d.get(pillar, {})
        if not _real_items(pd.get("highlights")):
            issues.append(f"{pillar} 面向缺少亮點分析")
        if not _real_items(pd.get("weaknesses")):
            issues.append(f"{pillar} 面向缺少改善建議")

    # 檢查每個面向的策略數量（只看總數會漏掉「某面向 0 條」）
    by_pillar = {"E": 0, "S": 0, "G": 0}
    for st in s.get("strategies", []):
        p = st.get("pillar") if isinstance(st, dict) else st.pillar
        if p in by_pillar:
            by_pillar[p] += 1
    for p, n in by_pillar.items():
        if n < 3:
            issues.append(f"{p} 面向策略不足（{n} 條，至少 3 條）")

    # 記錄到 trace（必須在 issues 算完之後；metadata 值統一用字串）
    get_client().update_current_span(
        metadata={"retry_count": str(state["retry_count"]),
                  "issues": "; ".join(issues) or "none",
                  "qc_passed": str(not issues)}
    )

    if issues:
        print(f"  ⚠️  QC 發現 {len(issues)} 個問題：{issues}")
        strategy_only = all("策略" in i for i in issues)
        return {"qc_passed": False,
                "retry_count": state["retry_count"] + 1,
                "retry_target": "solution" if strategy_only else "interpret",
                "errors": state["errors"] + issues}
    else:
        print("  ✅ QC 通過")
        return {"qc_passed": True, "errors": state["errors"]}


# ── Node 5：Generate Report ────────────────────────────────────────
@observe(capture_input=False)
def node_report(state: ESGState) -> dict:
    print(f"[4/4] 生成最終報告")
    company = state["company"]
    d = state.get("diagnosis", {})
    s = state.get("solution", {})

    lines = [
        f"# {company} ESG 分析報告",
        f"**來源報告**：{state['file_name']}",
        f"**分析時間**：{time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 整體摘要",
        d.get("overall_summary", "（無整體摘要）"),
        "",
    ]

    # 未通過 QC 時，在報告開頭明確標示
    if not state.get("qc_passed"):
        lines[3:3] = ["> ⚠️ **本報告未通過品質檢查，內容可能不完整，請勿直接採用。**"]

    pillar_emoji = {"E": "🌿", "S": "👥", "G": "⚖️"}
    pillar_full  = {"E": "環境（E）", "S": "社會（S）", "G": "治理（G）"}

    for pillar in ["E", "S", "G"]:
        pd = d.get(pillar, {})
        lines += [
            f"## {pillar_emoji[pillar]} {pillar_full[pillar]}",
            "",
            "**亮點 Highlights**",
        ]
        for h in pd.get("highlights", ["（無資料）"]):
            lines.append(f"- {_item_line(h)}")
        lines += ["", "**待改善 Weaknesses**"]
        for w in pd.get("weaknesses", ["（無資料）"]):
            lines.append(f"- {_item_line(w)}")
        kpis = pd.get("kpis", [])
        if kpis:
            lines += ["", "**關鍵績效指標 KPIs**"]
            for k in kpis[:10]:
                if isinstance(k, dict):
                    name, value, unit = (
                        k.get('name') or '',
                        k.get('value') or '',
                        k.get('unit') or '',)
                    lines.append(f"- {name}: {value} {unit}"
                                 + citations_to_markdown(k.get('sources', [])))
        lines.append("")

    lines += ["---", "", "## 📋 改善策略建議", ""]
    term_order = {"short": "🟢 短期（1年內）", "mid": "🟡 中期（2-3年）", "long": "🔵 長期（3-5年）"}
    grouped: dict = {"short": [], "mid": [], "long": []}
    for st in s.get("strategies", []):
        t = st.get("term", "short") if isinstance(st, dict) else st.term
        grouped[t].append(st)

    for term, label in term_order.items():
        if grouped[term]:
            lines.append(f"### {label}")
            for st in grouped[term]:
                if isinstance(st, dict):
                    p, a, r = st.get("pillar",""), st.get("action",""), st.get("rationale","")
                    srcs = st.get("sources", [])
                else:
                    p, a, r = st.pillar, st.action, st.rationale
                    srcs = st.citations
                lines.append(f"- **[{p}]** {a}" + citations_to_markdown(srcs)
                             + (f"  \n  _{r}_" if r else ""))
            lines.append("")

    lines += ["---", "", "## 🏆 同產業最佳實踐", ""]
    for bp in s.get("best_practices", []):
        if isinstance(bp, dict):
            p, sug, src = bp.get("pillar",""), bp.get("suggestion",""), bp.get("source","")
        else:
            p, sug, src = bp.pillar, bp.suggestion, bp.source
        lines += [f"**[{p}]** {sug}", f"> 來源：{src}", ""]

    lines += ["---", "", "## 🔍 GRI/TCFD 對標缺口分析", ""]
    for gap in s.get("gri_gaps", []):
        lines.append(f"- {gap}")

    if state.get("errors"):
        lines += ["", "---", "", "## ⚠️ 執行紀錄", ""]
        for e in state["errors"]:
            lines.append(f"- {e}")

    report_md = "\n".join(lines)
    fname = f"{company}_ESG分析報告_{time.strftime('%Y%m%d_%H%M')}.md"
    out_path = OUTPUTS_DIR / fname
    out_path.write_text(report_md, encoding="utf-8")

    # 同時存 JSON（供後續程式化處理）
    json_path = OUTPUTS_DIR / fname.replace(".md", ".json")
    json_path.write_text(json.dumps({
        "company": company, "diagnosis": d, "solution": s,
        "errors": state["errors"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  📝 報告已儲存：{out_path}")
    return {"report_path": str(out_path)}


# ── Routing functions ──────────────────────────────────────────────
def route_after_qc(state: ESGState) -> str:
    if state["qc_passed"]:
        return "report"
    # 上游失敗（找不到檔案等）重試無意義，達到上限也不再重試
    if _search_failed(state) or state["retry_count"] >= 2:
        print("  ⚠️  不再重試，直接輸出報告（內容可能不完整）")
        return "report"
    # 只有策略不足 → 只重跑 solution（省下重跑 interpret 的約 2.5 分鐘）
    return state.get("retry_target", "interpret")


# ── 建立 LangGraph ─────────────────────────────────────────────────
def build_graph():
    g = StateGraph(ESGState)
    g.add_node("search",   node_search)
    g.add_node("interpret", node_interpret)
    g.add_node("solution",  node_solution)
    g.add_node("qc",        node_quality_check)
    g.add_node("report",    node_report)

    g.add_edge(START,       "search")
    g.add_edge("search",    "interpret")
    g.add_edge("interpret", "solution")
    g.add_edge("solution",  "qc")
    g.add_conditional_edges("qc", route_after_qc,
                             {"report": "report", "interpret": "interpret",
                              "solution": "solution"})
    g.add_edge("report",    END)
    return g.compile()


# ── CLI 進入點 ────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ESG 多代理分析系統（LangGraph）")
    ap.add_argument("--file",    required=True, help="data/reports/ 中的 PDF 檔名")
    ap.add_argument("--company", default=None,  help="公司名稱（預設從檔名推斷）")
    ap.add_argument("--model",   default="ollama/gemma4-plan:latest", help="LiteLLM 模型名稱")
    ap.add_argument("--rebuild", action="store_true", help="重建 ChromaDB 索引")
    args = ap.parse_args()

    esg_utils.load_env = lambda: None  # 已在 call_llm 內部處理
    from dotenv import load_dotenv
    load_dotenv(esg_utils.PROJECT_ROOT / ".env")

    company = args.company or Path(args.file).stem.split("_")[0]

    graph = build_graph()
    initial_state: ESGState = {
        "file_name":   args.file,
        "company":     company,
        "model":       args.model,
        "rebuild":     args.rebuild,
        "sections":    [],
        "diagnosis":   {},
        "solution":    {},
        "retry_count": 0,
        "qc_passed":   False,
        "retry_target": "interpret",
        "errors":      [],
        "report_path": "",
    }

    print(f"\n🚀 ESG Agent 系統啟動")
    print(f"   公司：{company}")
    print(f"   報告：{args.file}")
    print(f"   模型：{args.model}")
    print()

    @observe(name="esg_pipeline", capture_output=False)
    def run_pipeline(state):
        # 把公司、模型、檔名標在 trace 上，方便在 Langfuse 篩選
        # （v4 以後 update_current_trace 已移除，改用 propagate_attributes）
        with propagate_attributes(
            tags=[company],
            metadata={"model": args.model, "file": args.file},
        ):
            return graph.invoke(state)

    t0 = time.time()
    final = run_pipeline(initial_state)
    get_client().flush()   # 短命的 CLI 程式一定要 flush，否則資料可能來不及送出

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"✅ 分析完成，耗時 {elapsed:.0f} 秒")
    print(f"📄 報告路徑：{final['report_path']}")
    if final["errors"]:
        print(f"⚠️  執行過程中有 {len(final['errors'])} 個警告，詳見報告末尾")
