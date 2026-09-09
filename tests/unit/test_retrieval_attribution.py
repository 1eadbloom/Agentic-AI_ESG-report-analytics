"""T033 — ChromaDB 檢索分支的 citation 完整性。

覆蓋 run_interpretation_agent 的「無 sections（T025 檢索）分支」：
fake ChromaDB collection 只回傳 documents/metadatas（模擬真實佈署——
section_id 存於 ChromaDB `id` 而非 metadata），
驗證每個非 unattributed Citation 的 chunk_id 都非空、且為原片段的 section_id，
同時 report↔KB 隔離在檢索路徑下依然成立（FR-002 / FR-004 / SC-001 / SC-002）。
"""
from types import SimpleNamespace

import interpretation_agent as ia

KB_TEXT = "GRI 標準要求揭露排放盤查邊界與方法學。"


def _mk(section_id: str, text: str, pillar: str, doc_type: str = "report",
        **kw) -> SimpleNamespace:
    base = dict(
        title="測試章節", text=text, pillar=pillar,
        source_file="A_2024.txt", page_range="p1-10",
        company="A公司", report_year="2024", doc_type=doc_type,
    )
    base.update(kw)
    return SimpleNamespace(section_id=section_id, **base)


def _sections():
    return [
        _mk("A_2024.txt::0",
            "A公司已完成 2024 年度溫室氣體盤查並取得第三方確信，"
            "排放量為 1200 公噸，較前年下降 12%，再生能源使用比例提升至 15%。",
            "E"),
        _mk("A_2024.txt::1",
            "A公司 2024 年員工教育訓練總時數達 35000 小時，女性主管比例 35%，"
            "並依 ISO 45001 建立職業安全衛生管理系統。",
            "S"),
        _mk("A_2024.txt::2",
            "A公司董事會治理架構完整，獨立董事 3 席占比達 1/3，"
            "並設置 ESG 委員會與風險管理架構。",
            "G"),
        # 知識庫片段（doc_type == knowledge_base）不能混入公司結論
        _mk("gri_standards_overview.md::0", KB_TEXT, "E",
            doc_type="knowledge_base"),
    ]


class _FakeCollection:
    """模擬 ChromaDB：section_id 只放在 `ids`（與 search_agent 一致），
    metadata 內**沒有** section_id。按 query_text 中的面向關鍵字回傳。"""

    def __init__(self, sections):
        self._sections = sections

    def _pillar_of(self, query: str) -> str:
        if "環境" in query:
            return "E"
        if "社會" in query:
            return "S"
        if "治理" in query:
            return "G"
        return "O"

    def query(self, query_texts, n_results=12, where=None, include=None):
        p = self._pillar_of(query_texts[0])
        secs = [s for s in self._sections if s.pillar == p]
        return {
            "ids": [[s.section_id for s in secs]],
            "documents": [[s.text for s in secs]],
            "metadatas": [[{
                "title": s.title, "pillar": s.pillar,
                "source_file": s.source_file, "page_range": s.page_range,
                "company": s.company, "report_year": s.report_year,
                "doc_type": s.doc_type,
            } for s in secs]],
        }


class _FakeLLM:
    def __call__(self, prompt, model=None, **kw):
        if "環境" in prompt:
            return ("【亮點 Highlights】\n"
                    "- 已完成 2024 年度溫室氣體盤查並取得第三方確信\n\n"
                    "【待改善 Weaknesses】\n"
                    "- 再生能源使用比例仍待提升\n\n"
                    "【關鍵績效指標 KPIs】\n"
                    "- 溫室氣體排放量：1200 公噸\n\n"
                    "【面向總結】\n整體表現良好。")
        if "社會" in prompt:
            return ("【亮點 Highlights】\n"
                    "- 建立 ISO 45001 職業安全衛生管理系統\n\n"
                    "【待改善 Weaknesses】\n"
                    "- 員工申訴案件處理時程未公告\n\n"
                    "【關鍵績效指標 KPIs】\n"
                    "- 教育訓練總時數：35000 小時\n\n"
                    "【面向總結】\n整體表現良好。")
        return ("【亮點 Highlights】\n"
                "- 獨立董事占比達 1/3\n\n"
                "【待改善 Weaknesses】\n"
                "- 董事 ESG 訓練時數未揭露\n\n"
                "【關鍵績效指標 KPIs】\n"
                "- 獨立董事席次：3 席\n\n"
                "【面向總結】\n整體表現良好。")


def _all_citations(diag):
    for pillar in ["E", "S", "G"]:
        pa = getattr(diag, pillar)
        for item in pa.highlights + pa.weaknesses:
            for c in item.citations:
                yield c
        for k in pa.kpis:
            for c in k.citations:
                yield c


def test_retrieval_path_citations_have_chunk_id(patch_embedding, monkeypatch):
    """FR-002：檢索路徑產出的 Citation 必須有非空 chunk_id。"""
    collection = _FakeCollection(_sections())
    monkeypatch.setattr(ia, "get_collection", lambda ef: (None, collection))
    monkeypatch.setattr(ia, "call_llm", _FakeLLM())

    diag = ia.run_interpretation_agent(
        company="A公司", source_file="A_2024.txt", sections=None)

    real = [c for c in _all_citations(diag) if not c.is_unattributed]
    assert real, "檢索路徑應產生至少一條可溯源 citation"

    known_ids = {"A_2024.txt::0", "A_2024.txt::1", "A_2024.txt::2"}
    for c in real:
        assert c.chunk_id, f"chunk_id 不得為空: {c.to_dict()}"
        assert c.chunk_id in known_ids, f"chunk_id 應為原 section_id: {c.to_dict()}"


def test_retrieval_path_isolation(patch_embedding, monkeypatch):
    """FR-004 / SC-002：檢索路徑不引入跨公司 / 跨年度 / KB 污染。"""
    collection = _FakeCollection(_sections())
    monkeypatch.setattr(ia, "get_collection", lambda ef: (None, collection))
    monkeypatch.setattr(ia, "call_llm", _FakeLLM())

    diag = ia.run_interpretation_agent(
        company="A公司", source_file="A_2024.txt", sections=None)

    real = [c for c in _all_citations(diag) if not c.is_unattributed]
    for c in real:
        assert c.company == "A公司", f"跨公司污染: {c.to_dict()}"
        assert c.report_year == "2024", f"跨年度污染: {c.to_dict()}"
        assert c.doc_type == "report", f"KB 混入結論: {c.to_dict()}"
        assert c.source_file == "A_2024.txt"