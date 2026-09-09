"""T023 驗證：三層 metadata 隔離整合測試（SC-002）。

以合成片段（兩家公司、兩個年度、報告 + 知識庫）驅動 run_interpretation_agent，
斷言輸出裡零跨公司 / 跨年度 / report↔KB 混用。
"""
import interpretation_agent as ia
from conftest import make_section

A_E = "A公司導入ISO50001能源管理系統，並完成溫室氣體盤查，排放量下降12%。"
A_S = "A公司2024年員工教育訓練總時數達20000小時，並建立職安衛管理系統。"
A_G = "A公司董事會設置獨立董事3席，並設審計委員會與永續發展委員會。"
B_E = "B公司2023年再生能源比例提升至25%，並承諾淨零目標。"
B_S = "B公司2023年推動多元共融政策，女性主管比例達40%。"
B_G = "B公司2023年強化風險管理與內部控制制度。"


def build_mixed_sections():
    A = [
        make_section(section_id="A::e1", company="A公司", report_year="2024",
                     pillar="E", text=A_E),
        make_section(section_id="A::s1", company="A公司", report_year="2024",
                     pillar="S", text=A_S),
        make_section(section_id="A::g1", company="A公司", report_year="2024",
                     pillar="G", text=A_G),
    ]
    B = [
        make_section(section_id="B::e1", company="B公司", report_year="2023",
                     pillar="E", text=B_E),
        make_section(section_id="B::s1", company="B公司", report_year="2023",
                     pillar="S", text=B_S),
        make_section(section_id="B::g1", company="B公司", report_year="2023",
                     pillar="G", text=B_G),
    ]
    KB = [
        make_section(section_id="kb::e", company="知識庫", report_year="2024",
                     pillar="E", doc_type="knowledge_base",
                     text="GRI 305 要求組織揭露排放在與減碳行動。"),
    ]
    return A + B + KB


class _FakeLLM:
    """依 prompt 回傳對應面向的結果，文字重用 A 公司片段詞彙以利歸因。"""

    def __call__(self, prompt, model=None, **kw):
        if "環境（Environmental）" in prompt:
            return ("【亮點 Highlights】\n"
                    "- A公司導入ISO50001能源管理系統並完成溫室氣體盤查\n\n"
                    "【待改善 Weaknesses】\n"
                    "- 減碳路徑揭露不足\n\n"
                    "【關鍵績效指標 KPIs】\n"
                    "- 溫室氣體排放量：12000 公噸\n\n"
                    "【面向總結】\nA公司環境揭露穩健")
        if "社會（Social）" in prompt:
            return ("【亮點 Highlights】\n"
                    "- A公司建立職安衛管理系統並持續員工訓練\n\n"
                    "【待改善 Weaknesses】\n"
                    "- 多元指標揭露不足\n\n"
                    "【關鍵績效指標 KPIs】\n"
                    "- 員工訓練時數：20000 小時\n\n"
                    "【面向總結】\nA公司社會面向穩定")
        return ("【亮點 Highlights】\n"
                "- A公司設置獨立董事並健全審計委員會\n\n"
                "【待改善 Weaknesses】\n"
                "- 治理指標完整性待補強\n\n"
                "【關鍵績效指標 KPIs】\n"
                "- 獨立董事席次：3 席\n\n"
                "【面向總結】\nA公司治理架構完善")


def _all_sources(diag: ia.DiagnosisResult):
    for p in ["E", "S", "G"]:
        pa = getattr(diag, p)
        for item in pa.highlights + pa.weaknesses:
            yield from item.citations
        for k in pa.kpis:
            yield from k.citations


def test_zero_cross_company_contamination(patch_embedding, monkeypatch):
    monkeypatch.setattr(ia, "call_llm", _FakeLLM())
    diag = ia.run_interpretation_agent(
        company="A公司", source_file="A_2024.txt", sections=build_mixed_sections())

    for c in _all_sources(diag):
        if c.is_unattributed:
            continue
        assert c.company == "A公司", f"跨公司污染: {c.to_dict()}"
        assert c.report_year == "2024", f"跨年度污染: {c.to_dict()}"
        assert c.doc_type == "report", f"KB 混入: {c.to_dict()}"


def test_every_item_has_sources(patch_embedding, monkeypatch):
    monkeypatch.setattr(ia, "call_llm", _FakeLLM())
    diag = ia.run_interpretation_agent(
        company="A公司", source_file="A_2024.txt", sections=build_mixed_sections())
    for p in ["E", "S", "G"]:
        pa = getattr(diag, p)
        for item in pa.highlights + pa.weaknesses:
            assert item.citations, f"{p} 的結論缺來源: {item.text}"
        for k in pa.kpis:
            assert k.citations, f"{p} 的 KPI 缺來源: {k.name}"


def test_kpi_exact_provenance_in_scope(patch_embedding, monkeypatch):
    """規則式 KPI 精確引用同公司同年度同類型的片段。"""
    monkeypatch.setattr(ia, "call_llm", _FakeLLM())
    diag = ia.run_interpretation_agent(
        company="A公司", source_file="A_2024.txt", sections=build_mixed_sections())
    for p in ["E", "S", "G"]:
        pa = getattr(diag, p)
        for k in pa.kpis:
            for c in k.citations:
                assert c.company == "A公司" and c.report_year == "2024"