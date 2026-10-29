"""解析回歸測試：鎖住在真實報告上發現的問題（垃圾 KPI、標題殘渣、策略格式）。"""
import pytest

import interpretation_agent as ia
import solution_agent as sa
from citation import AnalysisItem
from interpretation_agent import DiagnosisResult, PillarAnalysis


# ── 條目解析（_parse_bullets）─────────────────────────────────────
def test_bullets_drop_heading_debris_notes_and_fragments():
    block = (
        "- **排放減量成果的量化完整度**：缺乏年度減量百分比，難以評估成效。\n"
        "###\n"
        "---\n"
        "算\n"
        "  - *（註：相較於 GRI 401 員工留任率之標準）*\n"
        "- **範疇三的具體數據挑戰**：身為證券商，範疇三是最大的環境挑戰。\n"
    )
    items = ia._parse_bullets(block)
    assert len(items) == 2
    assert all("###" not in t and "註" not in t for t in items)
    assert not any(t.strip() == "算" for t in items)


def test_bullets_keep_unmarked_new_item_after_complete_sentence():
    """上一條以句號收尾、下一行沒有項目符號也沒縮排 → 是新條目，不能併成一行。"""
    block = (
        "- **性別多元化仍有提升空間**：七席董事中僅有一名女性獨立董事。\n"
        "決策透明度的量化揭露不足（針對 GRI 205）：未見實際案件數。\n"
    )
    assert len(ia._parse_bullets(block)) == 2


def test_bullets_merge_indented_continuation():
    block = ("- **第三項待改善**：說明文字足夠長度的內容\n"
             "  縮排的延續說明仍應併回上一條\n")
    assert len(ia._parse_bullets(block)) == 1


# ── KPI 驗證 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("name,value", [
    ("• 進行「季檢視」及「", "360 度"),               # 殘缺句、引號不成對
    ("分別為永康", "1 件"),                            # 連接詞開頭
    ("e) 及", "850 公噸"),                             # 括號不成對
    ("2：相關產業定義：➀ 燃煤開採：營收超逾", "25 %"),   # 政策門檻，不是績效
    ("金控1", "00 %"),                                 # 把 100% 從中間切開
    ("承銷、諮詢和證券化交易件數及金額", "需參照 FN-IB.001.A 之實際報告數據"),
    ("董事會出席率", "[文本片段數值不完整] %"),
    ("提供轉型計畫之國營事業/當地政府持股大於", "50 %"),   # 政策排除門檻
    ("自銷售及開採油砂活動之營收大於總營收", "10 %"),
])
def test_junk_kpis_are_rejected(name, value):
    assert not ia._valid_kpi(ia._clean_kpi_name(ia._strip_md(name)),
                             ia._strip_md(value))


@pytest.mark.parametrize("name,value", [
    ("上市櫃股票及公司債投資財務碳排放量", "43,319 tCO2e"),
    ("證券商（高級）業務員證照取得人次", "526 人次"),
    ("無紙化節省用紙量", "約 918 萬張"),
    ("完訓率為", "100 %"),
])
def test_real_kpis_are_kept(name, value):
    assert ia._valid_kpi(ia._clean_kpi_name(ia._strip_md(name)), value)


def test_kpi_name_cleaning_and_llm_noise():
    assert ia._clean_kpi_name("完訓率皆為") == "完訓率"
    assert ia._clean_kpi_name("總排放量約為") == "總排放量"
    assert ia._clean_llm_text("減少 $tCO_2e$，386 人人次") == "減少 tCO2e，386 人次"


def test_rule_based_kpis_require_a_unit():
    """名稱裡的數字（FTSE4Good、GRI 2021）不可被當成 KPI。"""
    txt = "FTSE4Good 指標 GRI 2021 準則 員工人數達 3000 人，用電量 120 萬度"
    kpis = ia.extract_kpis(txt, "E")
    assert kpis and all(k.unit for k in kpis)
    assert not any("FTSE" in k.name or "GRI" in k.name for k in kpis)


# ── 策略解析與重試（solution_agent）───────────────────────────────
def test_strategy_parser_handles_markdown_numbering_and_label_echo():
    raw = (
        "1. **[G][短期]** 建立供應商分級：識別關鍵供應商\n"
        "- [G][短期] 行動：設計永續培訓工作坊：協助供應商理解需求\n"   # 模型照抄「行動：」
        "[G][中期] 導入數位化平台：強化追蹤\n"
        "[E][中期] 這行是別的面向：應被忽略\n"
        "[G][長期] 將績效納入續約機制：整合至採購決策\n"
        "[G][長期] 將績效納入續約機制：重複的行動\n"
    )
    out = sa._parse_strategy_lines(raw, "G", [])
    assert [s.action for s in out] == [
        "建立供應商分級", "設計永續培訓工作坊", "導入數位化平台", "將績效納入續約機制"]
    assert all(s.pillar == "G" for s in out)
    assert out[1].rationale.startswith("協助供應商")


def _diag():
    d = DiagnosisResult("示範公司", "x.pdf")
    for p in "ESG":
        setattr(d, p, PillarAnalysis(p, weaknesses=[AnalysisItem("一段足夠長度的待改善說明")]))
    return d


def _five(p):
    return "\n".join(f"[{p}][{t}] 策略{i}：理由{i}"
                     for i, t in enumerate(["短期", "短期", "中期", "中期", "長期"]))


def test_pillar_with_unparseable_output_falls_back_and_is_labeled(monkeypatch):
    import re

    def fake(prompt, model=None, max_tokens=None):
        p = re.search(r"\[([ESG])\]\[期程\]", prompt).group(1)
        return "無法解析的回應" if p == "S" else _five(p)

    monkeypatch.setattr(sa, "call_llm", fake)
    out = sa.generate_strategies(_diag())
    by = {p: [s for s in out if s.pillar == p] for p in "ESG"}
    assert len(by["E"]) == 5 and len(by["G"]) == 5
    assert by["S"] and all(s.rationale.endswith(sa._FALLBACK_NOTE) for s in by["S"])


def test_retry_reuses_successful_pillars_and_regenerates_only_missing(monkeypatch):
    import re
    calls = []

    def fake(prompt, model=None, max_tokens=None):
        p = re.search(r"\[([ESG])\]\[期程\]", prompt).group(1)
        calls.append(p)
        return _five(p)

    monkeypatch.setattr(sa, "call_llm", fake)
    existing = (
        [sa.StrategyItem("E", "short", f"e{i}", "r") for i in range(5)]
        + [sa.StrategyItem("S", "short", f"x{i}", "r" + sa._FALLBACK_NOTE) for i in range(3)]  # 備援即使有 3 條也不可沿用
        + [sa.StrategyItem("G", "short", f"g{i}", "r") for i in range(5)])
    out = sa.generate_strategies(_diag(), existing=existing)
    assert calls == ["S"]
    assert {p: sum(1 for s in out if s.pillar == p) for p in "ESG"} == {"E": 5, "S": 5, "G": 5}


def test_empty_best_practice_response_uses_labeled_fallback(monkeypatch):
    class FakeCol:
        def count(self): return 8
        def query(self, **k):
            return {"documents": [["知識庫片段"]],
                    "metadatas": [[{"source_file": "kb.md"}]]}

    monkeypatch.setattr(sa, "get_embedding_fn", lambda: None)
    monkeypatch.setattr(sa, "get_collection", lambda ef: (None, FakeCol()))
    monkeypatch.setattr(sa, "call_llm", lambda *a, **k: "")
    out = sa.get_best_practices(_diag())
    assert len(out) == 3
    assert all(b["suggestion"].strip() for b in out)               # 不可留空白
    assert all("預設範本" in b["source"] for b in out)              # 且要標示來源


def test_gri_gaps_are_topic_based():
    from interpretation_agent import KPIItem
    d = _diag()
    d.E.kpis = [KPIItem(name="溫室氣體排放量", value="100", unit="tCO2e", pillar="E")]
    gaps = sa.analyze_gri_gaps(d)
    assert not any("GRI 305" in g for g in gaps)      # 有排放 KPI → 不算缺口
    assert any("GRI 302" in g for g in gaps)          # 沒有能源 KPI → 列為缺口


def test_kpi_name_cut_by_line_wrap_is_not_extracted():
    """「占總溫⏎室氣體排放量97%」會抓到少了一個字的「室氣體排放量」，語意被改掉，必須捨棄。"""
    txt = "其中816 公噸來自於範疇二的電力使用，占總溫\n室氣體排放量97%。"
    assert not any("室氣體" in k.name for k in ia.extract_kpis(txt, "E"))


def test_kpi_after_a_normal_delimiter_is_still_extracted():
    txt = "2024 年範疇一與範疇二之溫室氣體，總排放量約為840 公噸二氧化碳當量。"
    names = [(k.name, k.value) for k in ia.extract_kpis(txt, "E")]
    assert any(v == "840" for _, v in names)


def test_connector_prefixed_names_are_rejected():
    assert not ia._valid_kpi("另外總排放量", "840 公噸")
    assert ia._valid_kpi("總排放量", "840 公噸")
