"""T009 / T004 驗證：Citation 與 AnalysisItem 的欄位與序列化（FR-002）。"""
from citation import (
    AnalysisItem, Citation, UNATTRIBUTED, assert_section_scope,
    make_citation, make_unattributed,
)
from conftest import make_section


def test_citation_fields():
    s = make_section(section_id="r.pdf::7", source_file="r.pdf",
                     page_range="p12-15")
    c = make_citation(s, excerpt="本公司已完成盤查作業")
    assert c.company == "示範公司"
    assert c.report_year == "2024"
    assert c.chunk_id == "r.pdf::7"
    assert c.page_range == "p12-15"
    assert c.source_file == "r.pdf"
    assert c.doc_type == "report"
    assert not c.is_unattributed


def test_citation_serialization_roundtrip():
    s = make_section(section_id="r.pdf::3", doc_type="report")
    c = make_citation(s, excerpt="節錄文字")
    d = c.to_dict()
    assert d["company"] == "示範公司"
    assert d["report_year"] == "2024"
    assert d["chunk_id"] == "r.pdf::3"
    # roundtrip 後欄位一致
    assert Citation.from_dict(d).to_dict() == d


def test_analysis_item_serialization():
    s = make_section(section_id="r.pdf::9")
    ai = AnalysisItem("某項結論", [make_citation(s, excerpt="摘錄")])
    d = ai.to_dict()
    assert d["text"] == "某項結論"
    assert d["sources"][0]["chunk_id"] == "r.pdf::9"
    ai2 = AnalysisItem.from_dict(d)
    assert ai2.text == "某項結論"
    assert ai2.citations[0].company == "示範公司"


def test_unattributed_marker():
    u = make_unattributed("示範公司", "2024")
    assert u.is_unattributed
    assert u.chunk_id == UNATTRIBUTED
    assert u.source_file == ""
    assert u.company == "示範公司" and u.report_year == "2024"
    assert "非文件依據" in u.excerpt


def test_assert_section_scope():
    s = make_section()
    assert_section_scope(s, "示範公司", "2024")  # 不拋出


def test_assert_section_scope_company_mismatch():
    s = make_section(company="別家公司")
    try:
        assert_section_scope(s, "示範公司", "2024")
    except ValueError:
        return
    raise AssertionError("company 不符時應拋出 ValueError")


def test_assert_section_scope_year_mismatch():
    s = make_section(report_year="2023")
    try:
        assert_section_scope(s, "示範公司", "2024")
    except ValueError:
        return
    raise AssertionError("年度不符時應拋出 ValueError")


def test_assert_section_scope_invalid_doc_type():
    s = make_section(doc_type="other")
    try:
        assert_section_scope(s, "示範公司", "2024")
    except ValueError:
        return
    raise AssertionError("doc_type 無效時應拋出 ValueError")