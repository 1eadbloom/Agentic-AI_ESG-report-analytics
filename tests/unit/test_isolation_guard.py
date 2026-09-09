"""T022 驗證：隔離守門單元測試（FR-004 / SC-002）。"""
from citation import UNATTRIBUTED, assert_section_scope, select_sources
from conftest import make_section

E_TEXT = "A公司導入ISO50001能源管理系統，綠色採購與碳排放熱點盤點完備。"


def test_scope_guard_company_mismatch():
    s = make_section(company="A公司", report_year="2024")
    assert_section_scope(s, "A公司", "2024")  # OK
    try:
        assert_section_scope(s, "B公司", "2024")
    except ValueError:
        return
    raise AssertionError("company 不符應拋出 ValueError")


def test_select_sources_excludes_other_company(patch_embedding):
    a = make_section(section_id="A::e1", company="A公司", report_year="2024",
                     text=E_TEXT)
    b = make_section(section_id="B::e1", company="B公司", report_year="2023",
                     text="B公司導入ISO50001能源管理系統，綠色採購盤點完備。")
    cites = select_sources("導入ISO50001能源管理系統", [a, b], "A公司", "2024")
    assert all(c.company == "A公司" for c in cites)
    assert all(c.chunk_id != "B::e1" for c in cites)


def test_select_sources_excludes_other_year(patch_embedding):
    y24 = make_section(section_id="A::24", company="A公司", report_year="2024",
                       text=E_TEXT)
    y23 = make_section(section_id="A::23", company="A公司", report_year="2023",
                       text=E_TEXT)
    cites = select_sources("導入ISO50001能源管理系統", [y24, y23],
                           "A公司", "2024")
    assert all(c.report_year == "2024" for c in cites)
    assert all(c.chunk_id != "A::23" for c in cites)


def test_select_sources_never_mixes_kb(patch_embedding):
    kb = make_section(section_id="kb::1", company="A公司", report_year="2024",
                      doc_type="knowledge_base", text=E_TEXT)
    rep = make_section(section_id="A::e1", company="A公司", report_year="2024",
                       doc_type="report", text=E_TEXT)
    cites = select_sources("導入ISO50001能源管理系統", [kb, rep], "A公司", "2024")
    assert cites and all(c.doc_type == "report" for c in cites)
    assert all(c.chunk_id != "kb::1" for c in cites)


def test_select_sources_only_kb_unattributed(patch_embedding):
    kb = make_section(section_id="kb::1", company="A公司", report_year="2024",
                      doc_type="knowledge_base", text=E_TEXT)
    cites = select_sources("導入ISO50001能源管理系統", [kb], "A公司", "2024")
    assert len(cites) == 1 and cites[0].chunk_id == UNATTRIBUTED