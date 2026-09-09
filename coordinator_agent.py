"""
coordinator_agent.py — 任務協調與品質控管代理（Coordinator Agent）

用 LangGraph 將四個 Agent 串接成有狀態的工作流程：
  START → search → interpret → solution → quality_check → report → END
                                              ↑ 若 QC 失敗 (最多 2 次重試)
                                              └──────────────────┘

用法：
  python coordinator_agent.py --file "2024年永續報告書_中_中信證券000616.pdf"
  python coordinator_agent.py --file xxx.pdf --rebuild   # 重建索引
  python coordinator_agent.py --file xxx.pdf --model ollama/gemma3:4b
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
from solution_agent       import run_solution_agent, SolutionResult
from citation            import citations_to_markdown


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
    errors:      list[str]
    # 最終輸出
    report_path: str


# ── Node 1：Search ─────────────────────────────────────────────────
def node_search(state: ESGState) -> dict:
    print("=" * 60)
    print(f"[1/4] Search Agent — 解析報告：{state['file_name']}")
    try:
        p = REPORTS_DIR / state["file_name"]
        if not p.exists():
            return {"errors": state["errors"] + [f"找不到檔案：{p}"],
                    "sections": []}
        secs = run_search_agent(files=[p], rebuild=state["rebuild"],
                                   company=state["company"], include_kb=True)
        return {"sections": [s.__dict__ if hasattr(s, "__dict__") else s
                              for s in secs],
                "errors": state["errors"]}
    except Exception as e:
        return {"errors": state["errors"] + [f"Search Agent 失敗：{e}"],
                "sections": []}


# ── Node 2：Interpret ──────────────────────────────────────────────
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
def node_solution(state: ESGState) -> dict:
    print(f"[3/4] Solution Agent — 生成改善策略")
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
        sol = run_solution_agent(dr, model=state["model"])
        return {"solution": sol.to_dict(), "errors": state["errors"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"errors": state["errors"] + [f"Solution Agent 失敗：{e}"],
                "solution": {}}


# ── Node 4：Quality Check ──────────────────────────────────────────
def node_quality_check(state: ESGState) -> dict:
    print(f"[QC]  品質驗證（第 {state['retry_count']+1} 次）")
    issues = []
    d = state.get("diagnosis", {})
    s = state.get("solution", {})

    # 檢查 E/S/G 三面向都有分析結果
    for pillar in ["E", "S", "G"]:
        pd = d.get(pillar, {})
        if not pd.get("highlights"):
            issues.append(f"{pillar} 面向缺少亮點分析")
        if not pd.get("weaknesses"):
            issues.append(f"{pillar} 面向缺少改善建議")

    # 檢查策略數量
    if len(s.get("strategies", [])) < 3:
        issues.append("策略建議數量不足（需至少 3 條）")

    if issues:
        print(f"  ⚠️  QC 發現 {len(issues)} 個問題：{issues}")
        return {"qc_passed": False,
                "retry_count": state["retry_count"] + 1,
                "errors": state["errors"] + issues}
    else:
        print("  ✅ QC 通過")
        return {"qc_passed": True, "errors": state["errors"]}


# ── Node 5：Generate Report ────────────────────────────────────────
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
            for k in kpis[:5]:
                if isinstance(k, dict):
                    name, value, unit = (k.get('name', ''), k.get('value', ''),
                                         k.get('unit', ''))
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
    if state["retry_count"] >= 2:
        print("  ⚠️  已達最大重試次數，強制輸出報告")
        return "report"
    return "interpret"   # 重試：重新跑 interpret → solution → qc


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
                             {"report": "report", "interpret": "interpret"})
    g.add_edge("report",    END)
    return g.compile()


# ── CLI 進入點 ────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ESG 多代理分析系統（LangGraph）")
    ap.add_argument("--file",    required=True, help="data/reports/ 中的 PDF 檔名")
    ap.add_argument("--company", default=None,  help="公司名稱（預設從檔名推斷）")
    ap.add_argument("--model",   default="ollama/gemma3:4b", help="LiteLLM 模型名稱")
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
        "errors":      [],
        "report_path": "",
    }

    print(f"\n🚀 ESG Agent 系統啟動")
    print(f"   公司：{company}")
    print(f"   報告：{args.file}")
    print(f"   模型：{args.model}")
    print()

    t0 = time.time()
    final = graph.invoke(initial_state)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"✅ 分析完成，耗時 {elapsed:.0f} 秒")
    print(f"📄 報告路徑：{final['report_path']}")
    if final["errors"]:
        print(f"⚠️  執行過程中有 {len(final['errors'])} 個警告，詳見報告末尾")
