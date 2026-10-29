"""數值感知的引用歸因（回歸測試）。

背景：KPI 原本只用語意相似度挑引用，常引到「主題相近但沒有該數字」的片段
（例：526 人次實際在 p63 表格，卻引到 p102）。這裡鎖住修正後的行為，
以及實作時踩到的坑（表格儲存格一格一行，不能把換行與空白全部刪掉）。
"""
import pytest

from citation import (
    contains_number, select_sources, select_sources_for_value,
    value_found_in_sections, value_tokens,
)
from conftest import make_section


# ── 數字比對 ──────────────────────────────────────────────────────
def test_table_cells_on_separate_lines_are_not_glued():
    """表格擷取成一格一行：43,319／81,957／66,827 不可黏成一串而比對不到。"""
    table = "財務碳排放量\ntCO2e\n43,319\n81,957\n66,827\n"
    assert contains_number(table, "43319")
    assert contains_number("5.51\n9.37\n7.60", "5.51")


def test_number_boundaries():
    assert not contains_number("共 238 人", "38")        # 38 不可比對到 238
    assert not contains_number("共 380 人", "38")
    assert contains_number("占比約 38％", "38")           # 全形百分號
    assert contains_number("總計 1,234,567 元", "1234567")


def test_value_tokens_skip_years_and_single_digits():
    toks = value_tokens("2024年 1/7 約 14.3 % 43,319 tCO2e")
    assert toks == ["14.3", "43319"]
    assert value_tokens("1/7") == []                      # 沒有可比對的數字


# ── 引用歸因 ──────────────────────────────────────────────────────
def _sections():
    topic = make_section(          # 主題最像、但沒有數字
        section_id="r::topic", page_range="p102",
        text="證券商業務員證照取得人次持續成長，公司鼓勵員工取得專業證照並提供補助。")
    table = make_section(          # 真正含有數字的表格
        section_id="r::table", page_range="p63",
        text="證照類別\n通過人次\n526\n386\n")
    other = make_section(
        section_id="r::other", page_range="p9",
        text="本公司導入再生能源並完成溫室氣體盤查。")
    return [topic, table, other]


def test_kpi_citation_points_to_chunk_that_contains_the_value(patch_embedding):
    cites = select_sources_for_value(
        "證券商業務員證照取得人次: 526 人次", "526 人次", _sections(),
        "示範公司", "2024")
    assert [c.chunk_id for c in cites][0] == "r::table"
    assert all(not c.is_unattributed for c in cites)


def test_kpi_citation_only_uses_chunks_containing_value_when_available(patch_embedding):
    cites = select_sources_for_value(
        "證照取得人次: 526 人次", "526 人次", _sections(), "示範公司", "2024")
    assert {c.chunk_id for c in cites} == {"r::table"}


def test_value_not_in_text_falls_back_to_similarity(patch_embedding):
    """LLM 自己算的數字（例：3/7≈42.86%）原文找不到：退回語意歸因，且能被偵測出來。"""
    secs = _sections()
    assert value_found_in_sections("約 42.86 %", secs, "示範公司", "2024") is False
    assert value_found_in_sections("526 人次", secs, "示範公司", "2024") is True
    assert value_found_in_sections("1/7", secs, "示範公司", "2024") is None
    cites = select_sources_for_value("證照取得人次: 約 42.86 %", "約 42.86 %",
                                     secs, "示範公司", "2024")
    assert len(cites) >= 1          # 不拋例外；可能是相似段落或 unattributed


def test_select_sources_gives_number_bonus(patch_embedding):
    """結論含數字時，含該數字的片段要能被選入並排在前面。"""
    cites = select_sources("證照取得人次達 526 人次", _sections(), "示範公司", "2024")
    ids = [c.chunk_id for c in cites]
    assert "r::table" in ids
    assert ids[0] == "r::table"


def test_scope_isolation_still_enforced_for_value_attribution(patch_embedding):
    """跨公司的片段即使含有相同數字，也不能被引用。"""
    foreign = make_section(section_id="x::f", company="別家公司", text="通過人次 526")
    cites = select_sources_for_value("人次: 526", "526 人次", [foreign],
                                     "示範公司", "2024")
    assert len(cites) == 1 and cites[0].is_unattributed
