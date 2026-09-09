"""T012 + T029 驗證：JSON 輸出契約（contracts/output-json.md）。

含「sources 為 additive、非 unattributed 的欄位非空、excerpt ≤100、
隔離欄位、策略 sources ⊆ 弱點 sources」等契約斷言。
"""
from citation import Citation
from interpretation_agent import (
    AnalysisItem, DiagnosisResult, KPIItem, PillarAnalysis,
)
from solution_agent import SolutionResult, StrategyItem


def _cit(chunk_id, **kw):
    base = dict(company="示範公司", report_year="2024", source_file="r.pdf",
                chunk_id=chunk_id, page_range="p12-15", doc_type="report",
                excerpt="本公司已完成盤查作業，並透過第三方機構取得確信。")
    base.update(kw)
    return Citation(**base)


def _unattributed():
    return Citation(company="示範公司", report_year="2024", source_file="",
                    chunk_id="unattributed", page_range="", doc_type="",
                    excerpt="非文件依據 · 通用知識或經驗判斷，非特定文件片段")


def _pillar() -> PillarAnalysis:
    return PillarAnalysis(
        pillar="E",
        highlights=[AnalysisItem("盤查完成", [_cit("r.pdf::1")])],
        weaknesses=[
            AnalysisItem("減碳目標未承諾", [_cit("r.pdf::2")]),
            AnalysisItem("屬一般性說明", [_unattributed()]),
        ],
        kpis=[KPIItem(name="排放量", value="12000", unit="公噸", pillar="E",
                      citations=[_cit("r.pdf::1")])],
        raw_summary="環境面向表現穩健",
    )


def _walk_items(d):
    for pillar in ["E", "S", "G"]:
        pd = d[pillar]
        yield from pd["highlights"]
        yield from pd["weaknesses"]
        for k in pd["kpis"]:
            yield k


def test_json_contract_sources_present():
    diag = DiagnosisResult(company="示範公司", source_file="r.pdf",
                           E=_pillar()).to_dict()
    for item in _walk_items(diag):
        assert "sources" in item, f"item 缺 sources: {item}"
        assert item["sources"], f"item 缺來源: {item}"


def test_json_strategy_sources_present():
    sol = SolutionResult(
        company="示範公司",
        strategies=[StrategyItem(pillar="E", term="short", action="導入ISO50001",
                                 rationale="改善能源", citations=[_cit("r.pdf::2")])],
    ).to_dict()
    for st in sol["strategies"]:
        assert "sources" in st and st["sources"]


def test_json_non_unattributed_fields_nonempty():
    cit = _cit("r.pdf::1")
    d = cit.to_dict()
    for key in ["company", "report_year", "source_file", "chunk_id", "page_range"]:
        assert d[key], f"{key} 應為非空"
    assert len(d["excerpt"]) <= 100


def test_json_excerpt_in_sources(patch_embedding):
    """T029：sources 的 excerpt 非空（由歸因產生）。"""
    from interpretation_agent import analyze_pillar
    from conftest import make_section
    sec = make_section(
        section_id="r.pdf::1", pillar="E", text="本公司已完成盤查並取得第三方確信。")
    # 直接以無 LLM 空輸出驗證 excerpt 從歸因路徑產生
    from esg_utils import call_llm as _real_call  # noqa: F401

    class _FakeLLM:
        def __call__(self, prompt, model=None, **kw):
            return ("【亮點 Highlights】\n- 盤查完成並取得確信\n\n"
                    "【待改善 Weaknesses】\n- 減碳目標未承諾\n\n"
                    "【關鍵績效指標 KPIs】\n- 排放量：12000 公噸\n\n"
                    "【面向總結】\n整體穩健")
    import interpretation_agent as ia
    ia.call_llm = _FakeLLM()
    try:
        pa = ia.analyze_pillar("E", [sec], "示範公司", "2024")
    finally:
        import esg_utils
        ia.call_llm = esg_utils.call_llm
    item = pa.highlights[0]
    assert item.citations and item.citations[0].excerpt
    assert len(item.citations[0].excerpt) <= 100


def test_json_isolation_fields():
    diag = DiagnosisResult(company="示範公司", source_file="r.pdf",
                           E=_pillar()).to_dict()
    for item in _walk_items(diag):
        for src in item["sources"]:
            if src["chunk_id"] == "unattributed":
                continue
            assert src["company"] == "示範公司"
            assert src["report_year"] == "2024"
            assert src["doc_type"] == "report"


def test_strategy_sources_subset_of_weakness(patch_embedding):
    """契約不變式：策略 sources ⊆ 其面向弱點 sources。"""
    import json as _json
    from solution_agent import generate_strategies
    diag = DiagnosisResult(company="示範公司", source_file="r.pdf", E=_pillar())

    import solution_agent as sa

    class _FakeLLM:
        def __call__(self, prompt, model=None, **kw):
            if "永續發展改善策略" in prompt:
                return ("[E][短期] 導入ISO50001：改善能源管理\n"
                        "[E][中期] 設定SBTi：對應淨零趨勢\n")
            return "（無）"
    orig = sa.call_llm
    sa.call_llm = _FakeLLM()
    try:
        strategies = sa.generate_strategies(diag)
    finally:
        sa.call_llm = orig

    weak_sources = {_json.dumps(c.to_dict(), ensure_ascii=False, sort_keys=True)
                    for w in diag.E.weaknesses for c in w.citations}
    for st in strategies:
        for c in st.citations:
            key = _json.dumps(c.to_dict(), ensure_ascii=False, sort_keys=True)
            assert key in weak_sources, "策略引用了不屬於其弱點的來源"