"""Quickstart 第 3 步：E2E CLI run（mocked LLM）。

以離線方式驅動完整管線（interpret → solution → report rendering），
驗證 Markdown / JSON 契約與隔離保證。無需 PDF、無需 sentence-transformers、
無需 ChromaDB。
"""
import json

import coordinator_agent
import interpretation_agent as ia
import solution_agent as sa
from search_agent import ESGSection

COMPANY = "示範公司"
YEAR = "2024"
FILE = "example_report.txt"


def build_sections():
    return [
        ESGSection(section_id="ex::e1", title="環境", pillar="E",
                   text="本公司導入 ISO 50001 能源管理系統並完成溫室氣體盤查，"
                        "2024 年度排放量為 12000 公噸，較前年下降 12%，"
                        "再生能源使用比例為 15%。",
                   source_file=FILE, page_range="p3-4",
                   company=COMPANY, report_year=YEAR, doc_type="report"),
        ESGSection(section_id="ex::s1", title="社會", pillar="S",
                   text="本公司 2024 年員工教育訓練總時數達 35000 小時，女性主管"
                        "比例 35%，並依 ISO 45001 建立職安衛管理系統。",
                   source_file=FILE, page_range="p5-6",
                   company=COMPANY, report_year=YEAR, doc_type="report"),
        ESGSection(section_id="ex::g1", title="治理", pillar="G",
                   text="本公司董事會設置獨立董事 3 席，並設審計委員會與永續發展"
                        "委員會，將氣候風險納入風險管理架構。",
                   source_file=FILE, page_range="p7-8",
                   company=COMPANY, report_year=YEAR, doc_type="report"),
    ]


class _FakeLLM:
    def __call__(self, prompt, model=None, max_tokens=2048, **kw):
        if "永續發展改善策略" in prompt:
            return (
                "[E][短期] 導入 ISO 50001 能源管理系統並強化能源揭露：呼應能源管理缺口，短期可執行\n"
                "[E][中期] 設定 SBTi 科學基礎減碳目標：對應國際淨零趨勢\n"
                "[E][長期] 於 2050 年達成營運淨零：呼應長期淨零承諾\n"
                "[S][短期] 提高供應鏈揭露與職安衛指標完整性：補齊 S 面向揭露缺口\n"
                "[S][中期] 公告員工申訴案件處理時程與結果：強化勞資溝通透明度\n"
                "[G][短期] 量化揭露獨立董事 ESG 訓練時數：補齊治理揭露缺口\n"
                "[G][中期] 將 ESG 指標納入高階主管績效考核：連結治理與績效\n"
            )
        if "環境（Environmental）" in prompt:
            return (
                "【亮點 Highlights】\n"
                "- 本公司導入 ISO 50001 能源管理系統並完成溫室氣體盤查\n"
                "- 再生能源使用比例提升至 15%，排放量較前年下降 12%\n\n"
                "【待改善 Weaknesses】\n"
                "- 尚未承諾 SBTi 科學基礎減碳目標\n"
                "- 再生能源使用比例仍低於同業平均\n\n"
                "【關鍵績效指標 KPIs】\n"
                "- 溫室氣體排放量：12000 公噸\n\n"
                "【面向總結】\n環境面向揭露完整，盤查制度健全。"
            )
        if "社會（Social）" in prompt:
            return (
                "【亮點 Highlights】\n"
                "- 2024 年員工教育訓練總時數達 35000 小時，每人平均受訓 40 小時\n"
                "- 依 ISO 45001 建立職安衛管理系統\n\n"
                "【待改善 Weaknesses】\n"
                "- 員工申訴案件處理時程未公告\n\n"
                "【關鍵績效指標 KPIs】\n"
                "- 員工訓練總時數：35000 小時\n\n"
                "【面向總結】\n社會面向穩定。"
            )
        return (
            "【亮點 Highlights】\n"
            "- 董事會設置獨立董事 3 席，並設審計委員會與永續發展委員會\n\n"
            "【待改善 Weaknesses】\n"
            "- 反貪腐政策與風險評估揭露不足\n\n"
            "【關鍵績效指標 KPIs】\n"
            "- 獨立董事席次：3 席\n\n"
            "【面向總結】\n治理架構完善。"
        )


def test_end_to_end_with_citations(patch_embedding, monkeypatch, tmp_path):
    monkeypatch.setattr(ia, "call_llm", _FakeLLM())
    monkeypatch.setattr(sa, "call_llm", _FakeLLM())
    monkeypatch.setattr(sa, "get_best_practices",
                        lambda diagnosis, model=None: sa._fallback_best_practices())
    monkeypatch.setattr(coordinator_agent, "OUTPUTS_DIR", tmp_path)

    diag = ia.run_interpretation_agent(company=COMPANY, source_file=FILE,
                                       sections=build_sections())
    sol = sa.run_solution_agent(diag)

    state = {
        "file_name": FILE, "company": COMPANY, "model": "mock",
        "sections": [], "diagnosis": diag.to_dict(),
        "solution": sol.to_dict(), "retry_count": 0, "qc_passed": True,
        "errors": [], "report_path": "",
    }
    out = coordinator_agent.node_report(state)
    md = (tmp_path / (out["report_path"].replace("\\", "/").split("/")[-1]))
    md_text = md.read_text(encoding="utf-8")
    mdp = out["report_path"].replace(".md", ".json")
    js = json.loads((tmp_path / mdp.replace("\\", "/").split("/")[-1]).read_text(encoding="utf-8"))

    # ── JSON 契約 ──
    for pillar in ["E", "S", "G"]:
        pd = js["diagnosis"][pillar]
        for item in pd["highlights"] + pd["weaknesses"]:
            _assert_sources_valid(item, "conclusion")
        for k in pd["kpis"]:
            _assert_sources_valid(k, "kpi")
    for st in js["solution"]["strategies"]:
        _assert_sources_valid(st, "strategy")

    # 策略 sources ⊆ 其面向弱點 sources
    weak_map = {p: _sources_of(js["diagnosis"][p]["weaknesses"])
                for p in ["E", "S", "G"]}
    for st in js["solution"]["strategies"]:
        assert st["sources"], "策略缺少 sources"
        for s in st["sources"]:
            key = json.dumps(s, ensure_ascii=False, sort_keys=True)
            assert key in weak_map[st["pillar"]], \
                f"策略 {st['action']} 引用非該面向弱點來源: {s}"

    # ── Markdown 契約 ──
    lines = md_text.splitlines()
    for bul in _bullets_between(lines, "**亮點 Highlights**", "**待改善 Weaknesses**"):
        assert "來源:" in bul or "非文件依據" in bul, f"缺引用: {bul}"
    for bul in _bullets_between(lines, "**待改善 Weaknesses**", "**關鍵績效指標 KPIs**"):
        assert "來源:" in bul or "非文件依據" in bul, f"缺引用: {bul}"
    for bul in _bullets_between(lines, "**關鍵績效指標 KPIs**", "## "):
        assert "來源:" in bul or "非文件依據" in bul, f"缺引用: {bul}"
    for bul in _bullets_between(lines, "## 📋 改善策略建議", "## 🏆"):
        assert "來源:" in bul or "非文件依據" in bul, f"策略缺引用: {bul}"


def _sources_of(items):
    return {json.dumps(s, ensure_ascii=False, sort_keys=True)
            for it in items for s in it["sources"]}


def _assert_sources_valid(item, kind):
    assert "sources" in item and item["sources"], f"{kind} 缺 sources: {item}"
    for s in item["sources"]:
        if s["chunk_id"] == "unattributed":
            assert s["excerpt"], "unattributed 需有說明文字"
            continue
        for k in ["company", "report_year", "source_file", "chunk_id", "page_range"]:
            assert s[k], f"{kind} source 的 {k} 為空: {s}"
        assert s["company"] == COMPANY
        assert s["report_year"] == YEAR
        assert s["doc_type"] == "report"
        assert s["excerpt"], "非 unattributed 來源需有摘錄"
        assert len(s["excerpt"]) <= 100


def _bullets_between(lines, start, end):
    active = False
    out = []
    for ln in lines:
        if start in ln:
            active = True
            continue
        if active and end in ln:
            break
        if active and ln.startswith("- "):
            out.append(ln)
    return out