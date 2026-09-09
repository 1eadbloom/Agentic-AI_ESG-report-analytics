"""T010 驗證：確定性歸因（R1）——top-k 門檻選取、unattributed 標記（FR-005/FR-006）。"""
from citation import (
    UNATTRIBUTED, keyword_overlap, score_against_sections, select_sources,
)
from conftest import make_section

SEC1 = "本公司以綠色採購為核心，全面盤點零售供應鏈的能源使用效率與碳排放熱點。"
SEC2 = "本公司建置再生能源設備，再生能源發電量與綠電使用比例持續成長。"
SEC3 = "本公司制定水資源再利用策略，並成立節水專案小組持續推進節水目標。"


def sections():
    return [
        make_section(section_id="r.pdf::1", text=SEC1, pillar="E"),
        make_section(section_id="r.pdf::2", text=SEC2, pillar="E"),
        make_section(section_id="r.pdf::3", text=SEC3, pillar="E"),
    ]


def test_attribution_topk(patch_embedding):
    """FR-005：多來源結論列出所有貢獻片段。"""
    text = "本公司建置再生能源設備並制定水資源再利用策略，綠電使用比例持續成長。"
    cites = select_sources(text, sections(), "示範公司", "2024")
    ids = {c.chunk_id for c in cites}
    assert "r.pdf::2" in ids
    assert "r.pdf::3" in ids
    assert len(cites) >= 2
    for c in cites:
        assert not c.is_unattributed
        assert c.company == "示範公司" and c.report_year == "2024"


def test_threshold_unattributed(patch_embedding):
    """FR-006：低於門檻 → 單一 unattributed + 說明。"""
    text = "xyz unrelated qwerty zzz nnmm aabbccddee fffggg hhh iiii jjjjj kkk"
    cites = select_sources(text, sections(), "示範公司", "2024")
    assert len(cites) == 1
    assert cites[0].chunk_id == UNATTRIBUTED
    assert "非文件依據" in cites[0].excerpt


def test_score_order(patch_embedding):
    text = "本公司同意再生能源設備建置並擴大綠電使用比例。"
    scored = score_against_sections(text, sections())
    assert scored[0][0].section_id == "r.pdf::2"
    # 依分數降冪
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_keyword_overlap_fallback():
    """embedding 不可用時退回關鍵字重疊。"""
    s2 = make_section(section_id="r.pdf::2", text=SEC2)
    other = make_section(section_id="r.pdf::9", text=SEC1)
    t = "再生能源綠電使用比例"
    assert keyword_overlap(t, SEC2) > keyword_overlap(t, SEC1)
    _ = s2, other


def test_kpi_exact_provenance(patch_embedding):
    """R1：規則式 KPI 精確溯源到匹配的 section。"""
    from interpretation_agent import extract_kpis
    secs = [
        make_section(section_id="r.pdf::7", pillar="E",
                     text="本公司2024年度溫室氣體排放量為 12000 公噸，較前年下降12%。"),
        make_section(section_id="r.pdf::8", pillar="E",
                     text="再生能源使用比例提升至 15%。"),
    ]
    kpis = extract_kpis("", "E", sections=secs)
    by_name = {k.name: k for k in kpis}
    item = next((k for nn, k in by_name.items() if "排放量" in nn), None)
    assert item is not None
    assert item.citations, "規則式 KPI 必須有 citation"
    assert item.citations[0].chunk_id == "r.pdf::7"
    assert [k.citations for k in kpis if k.name == item.name]