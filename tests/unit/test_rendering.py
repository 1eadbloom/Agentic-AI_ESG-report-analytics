"""T011 + T020 驗證：Markdown 內嵌引用渲染（contracts/markdown-report.md）。"""
from citation import (
    Citation, citations_to_markdown, format_source_part, make_unattributed,
)

C1 = Citation(company="示範公司", report_year="2024", source_file="r.pdf",
              chunk_id="r.pdf::3", page_range="p12-15", doc_type="report",
              excerpt="本公司已完成 2024 年度溫室氣體盤查作業，並取得 ISO 14064 確信，"
                      "整體碳排放量較前一年度下降約 12%，再生能源使用比例逐年提升。")
C2 = Citation(company="示範公司", report_year="2024", source_file="r.pdf",
              chunk_id="r.pdf::5", page_range="p22-24", doc_type="report")


def test_render_single_source():
    s = citations_to_markdown([C1])
    assert s.startswith(" (來源: ")
    assert "示範公司 · 2024 · chunk r.pdf::3 · p12-15" in s
    # 第一個來源含摘錄（60 字截斷 + …）
    assert "：「" in s
    assert "…」" in s
    assert "pdftest" not in s


def test_render_multiple_sources():
    """FR-005/馬克down 契約：多來源以 ` ; ` 連接，只有第一個帶摘錄。"""
    s = citations_to_markdown([C1, C2])
    assert " ; " in s
    assert s.count("來源:") == 2
    # 第二個來源不含摘錄
    parts = s.split(" ; ")
    assert "「" not in parts[1]


def test_render_unattributed():
    u = make_unattributed("示範公司", "2024")
    s = citations_to_markdown([u])
    assert "非文件依據" in s
    assert "unattributed" not in s


def test_render_knowledge_base():
    kb = Citation(company="知識庫", report_year="2024", source_file="kb.md",
                  chunk_id="gri_standards_overview.md::2", page_range="p1",
                  doc_type="knowledge_base", excerpt="GRI 標準要求組織揭露邊界")
    s = citations_to_markdown([kb])
    assert "KB 來源" in s
    assert "chunk gri_standards_overview.md::2" in s


def test_format_source_part_excerpt_cap():
    long_excerpt = "甲" * 200
    c = Citation(company="X", report_year="2024", source_file="f", chunk_id="c::1",
                 page_range="p1", doc_type="report", excerpt=long_excerpt)
    part = format_source_part(c, include_excerpt=True)
    assert "甲" * 60 + "…" in part
    assert "甲" * 61 not in part


def test_citations_to_markdown_empty():
    assert citations_to_markdown([]) == ""