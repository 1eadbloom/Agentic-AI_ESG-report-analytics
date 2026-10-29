"""KPI 名稱與數值的語意鄰近檢查，以及選用的 LLM 檢查員（用假的 LLM 測試，完全離線）。"""
import json
from types import SimpleNamespace

import kpi_judge
from citation import (
    kpi_supported_by_citations, make_citation, value_near_name_in_texts,
)


# ── 鄰近檢查 ──────────────────────────────────────────────────────
def test_number_next_to_its_name_in_prose_is_supported():
    t = "本公司2024年範疇二溫室氣體排放量約為850公噸，占總排放量97%。"
    assert value_near_name_in_texts("範疇二溫室氣體排放量", "850 公噸", [t]) is True


def test_number_that_belongs_to_another_item_is_rejected():
    """真實案例：董事長室配置資源＝100%，其實 100% 是旁邊『完訓率』的數字。"""
    t = ("設定董事及員工年度誠信經營教育訓練完訓率為100%，深化誠信文化於日常作業之中。"
         "增加網站英文版內容並定期更新。")
    assert value_near_name_in_texts("董事長室配置資源", "100 %", [t]) is False


def test_table_cell_with_header_in_same_chunk_is_supported():
    t = "證照名稱\n人次\n證券商（高級）業務員\n526\n期貨商業務員\n386\n"
    assert value_near_name_in_texts("證券商業務員證照取得人次", "526 人次", [t]) is True


def test_table_cell_without_header_is_undecidable_not_rejected():
    """表格欄位名稱可能在相鄰 chunk：不能因此把正確的 KPI 判成錯。"""
    t = "2.39(符合)\n≦63%\n61%\n依資產類別顯示投資組合財務碳排放\n"
    assert value_near_name_in_texts("金控溫度分數標的無減碳目標佔比", "61 %", [t]) is None


def test_prose_occurrence_without_name_beats_table_occurrence():
    t = "完訓率為100%，深化誠信文化。\n其他資料\n100%\n"
    assert value_near_name_in_texts("董事長室配置資源", "100 %", [t]) is False


def test_undecidable_inputs_return_none():
    assert value_near_name_in_texts("人數", "526 人", ["526 人"]) is None        # 名稱不足 3 個中文字
    assert value_near_name_in_texts("董事會成員總數", "約略", ["七席董事"]) is None   # 數值沒有任何數字


def test_supported_by_citations_uses_the_cited_chunks_only():
    near = SimpleNamespace(section_id="r::1", company="示範公司", report_year="2024",
                           source_file="r.pdf", page_range="p3", doc_type="report",
                           text="範疇二溫室氣體排放量約為850公噸。")
    far = SimpleNamespace(section_id="r::2", company="示範公司", report_year="2024",
                          source_file="r.pdf", page_range="p9", doc_type="report",
                          text="其他說明 850 公噸的電力。範疇二溫室氣體排放量的定義如下。")
    cites_near = [make_citation(near)]
    assert kpi_supported_by_citations("範疇二溫室氣體排放量", "850 公噸",
                                      cites_near, [near, far]) is True
    assert kpi_supported_by_citations("範疇二溫室氣體排放量", "850 公噸", [], [near]) is None


# ── LLM 檢查員（假的 LLM）─────────────────────────────────────────
EVIDENCE = ["證照名稱\n人次\n證券商（高級）業務員\n526\n期貨商業務員\n386\n"]


def _fake(reply: dict | str):
    return lambda prompt: reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


def test_judge_accepts_a_verbatim_quote_containing_the_value():
    r = {"supported": True, "quote": "證券商（高級）業務員 526"}
    assert kpi_judge.judge_kpi("證券商業務員證照取得人次", "526 人次", EVIDENCE, call=_fake(r)) is True


def test_judge_rejects_a_fabricated_quote():
    """模型說支持、但引文不在原文裡：一律視為不支持。"""
    r = {"supported": True, "quote": "證券商業務員證照取得人次共526人次，較去年成長"}
    assert kpi_judge.judge_kpi("證券商業務員證照取得人次", "526 人次", EVIDENCE, call=_fake(r)) is False


def test_judge_rejects_a_quote_that_lacks_the_value():
    r = {"supported": True, "quote": "證照名稱 人次"}
    assert kpi_judge.judge_kpi("證券商業務員證照取得人次", "526 人次", EVIDENCE, call=_fake(r)) is False


def test_judge_accepts_table_fragments_joined_by_ellipsis():
    r = {"supported": True, "quote": "證券商（高級）業務員…526"}
    assert kpi_judge.judge_kpi("證券商業務員證照取得人次", "526 人次", EVIDENCE, call=_fake(r)) is True


def test_judge_false_and_unparseable():
    assert kpi_judge.judge_kpi("x", "526 人次", EVIDENCE,
                               call=_fake({"supported": False, "quote": ""})) is False
    assert kpi_judge.judge_kpi("x", "526 人次", EVIDENCE, call=_fake("我覺得是對的")) is None
    assert kpi_judge.judge_kpi("x", "526 人次", EVIDENCE,
                               call=_fake('好的：{"supported": true, "quote": "證券商（高級）業務員 526"}')) is True


def test_judge_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ESG_KPI_JUDGE", raising=False)
    assert kpi_judge.enabled() is False
    monkeypatch.setenv("ESG_KPI_JUDGE", "1")
    monkeypatch.delenv("ESG_JUDGE_MODEL", raising=False)
    assert kpi_judge.enabled() is False        # 沒設模型也不啟用


def test_judge_ignores_braces_inside_think_blocks():
    raw = ('<think>先想一下 {"supported": false} 這只是草稿</think>\n'
           '{"supported": true, "quote": "證券商（高級）業務員 526"}')
    assert kpi_judge.judge_kpi("證券商業務員證照取得人次", "526 人次", EVIDENCE,
                               call=lambda p: raw) is True


# ── 中文數字：原文寫「十日」，KPI 寫「10 天」 ──────────────────────
def test_chinese_numeral_with_unit_matches_arabic_value():
    from citation import contains_number
    t = "員工因陪伴其配偶妊娠產檢或分娩得申請陪產檢及陪產假共十 日，請假期間工資照給。"
    assert contains_number(t, "10")
    assert contains_number("董事會由七席董事組成，其中二十五人次出席", "25")


def test_chinese_numeral_normalization_does_not_touch_ordinary_words():
    from citation import _norm_for_numbers
    assert _norm_for_numbers("統一規範，十分重視") == "統一規範，十分重視"


def test_paid_leave_kpi_is_supported_for_the_right_reason():
    """真實案例：陪產假「共十日」。先前確定性檢查只是被別處的 10（SDG 10、10 分鐘）湊巧放行。"""
    near = "員工因陪伴其配偶妊娠產檢或分娩得申請陪產檢及陪產假共十 日，請假期間工資照給。"
    far = ("實現聯合國永續發展目標第5 項 (SDG 5) 性別平權與第10 項 (SDG 10) 減少不平等。"
           + "其他說明文字。" * 30 + "陪產假相關規定如下。")
    assert value_near_name_in_texts("陪產假天數", "10 天", [near]) is True
    assert value_near_name_in_texts("陪產假天數", "10 天", [far]) is False


# ── 檢查員只處理「無法判斷」的 KPI ────────────────────────────────
def _pillar_with_table_kpi(monkeypatch, near_value):
    import interpretation_agent as ia
    from conftest import make_section
    secs = [make_section(section_id="r::1", company="示範公司", pillar="E",
                         text="2024年財務碳排放量為 43,319 tCO2e。\n2.39(符合)\n≦63%\n61%\n依資產類別顯示投資組合\n")]
    answers = iter([
        "【亮點 Highlights】\n- 這是一段足夠長度的亮點說明文字內容\n"
        "【待改善 Weaknesses】\n- 這是一段足夠長度的待改善說明文字內容\n"
        "【關鍵績效指標 KPIs】\n- 範疇三投資組合盤查覆蓋率：38 %\n- 財務碳排放量：43,319 tCO2e\n- 溫度分數涵蓋比例：61 %\n"
        "【面向總結】\n整體表現尚可。"])
    monkeypatch.setattr(ia, "call_llm", lambda *a, **k: next(answers))
    return ia, secs


def test_judge_is_called_only_when_proximity_check_is_undecidable(monkeypatch, patch_embedding):
    ia, secs = _pillar_with_table_kpi(monkeypatch, None)
    asked = []
    monkeypatch.setattr(ia._judge, "enabled", lambda: True)
    monkeypatch.setattr(ia._judge, "judge_kpi",
                        lambda name, value, ev, call=None: (asked.append(name), True)[1])
    ia.analyze_pillar("E", secs, "示範公司", "2024")
    # 43,319 與名稱相鄰（確定性檢查已通過）→ 不必問；只有表格儲存格 61%、名稱不在同一片段的那一項才問
    assert asked == ["溫度分數涵蓋比例"]


def test_judge_rejection_drops_only_undecidable_kpi(monkeypatch, patch_embedding):
    ia, secs = _pillar_with_table_kpi(monkeypatch, None)
    monkeypatch.setattr(ia._judge, "enabled", lambda: True)
    monkeypatch.setattr(ia._judge, "judge_kpi", lambda *a, **k: False)
    pa = ia.analyze_pillar("E", secs, "示範公司", "2024")
    assert not any("溫度分數" in k.name for k in pa.kpis)


# ── 亂碼字元：重問一次 ─────────────────────────────────────────────
def test_garbled_response_is_regenerated_once(monkeypatch, patch_embedding):
    import interpretation_agent as ia
    from conftest import make_section
    secs = [make_section(section_id="r::1", company="示範公司", pillar="E",
                         text="2024年財務碳排放量 43,319 tCO2e。")]
    tmpl = ("【亮點 Highlights】\n- 氣候風險拆解為「轉型風險」與「{x}體風險」並建立治理架構\n"
            "【待改善 Weaknesses】\n- 這是一段足夠長度的待改善說明文字內容\n"
            "【關鍵績效指標 KPIs】\n- 財務碳排放量：43,319 tCO2e\n【面向總結】\n整體表現尚可。")
    import itertools
    clean = tmpl.replace("{x}", "實")
    answers = itertools.chain([tmpl.replace("{x}", "實\ufffd實")], itertools.repeat(clean))
    calls = []
    monkeypatch.setattr(ia, "call_llm", lambda *a, **k: (calls.append(1), next(answers))[1])
    pa = ia.analyze_pillar("E", secs, "示範公司", "2024")
    assert len(calls) >= 2                       # 第一次含亂碼 → 至少重問一次
    assert "\ufffd" not in pa.highlights[0].text


# ── 目標 vs 實績 ───────────────────────────────────────────────────
def test_context_is_target_distinguishes_commitment_from_result():
    from citation import context_is_target
    tgt = "設定董事及員工年度誠信經營教育訓練完訓率為100%，深化誠信文化"
    act = "合計有7名董事及688名員工完成訓練，完訓率為100%。"
    assert context_is_target(tgt, tgt.index("100"))
    assert not context_is_target(act, act.index("100"))
    horizon = "訂定短期（2025年）累計減碳18%、中期（2030年）累計減碳40.5%、長期（2035年）累計減碳63%。"
    assert context_is_target(horizon, horizon.index("63"))
    assert not context_is_target("2024年受檢率達87.4%，輔助金額達271萬元。", 8)


def test_kpi_whose_value_only_appears_as_a_target_is_flagged():
    from citation import value_is_target_in_texts
    t = "設定董事及員工年度誠信經營教育訓練完訓率為100%，深化誠信文化於日常作業之中。"
    assert value_is_target_in_texts("誠信經營教育訓練完訓率", "100 %", [t]) is True


def test_kpi_with_an_actual_occurrence_is_not_flagged():
    from citation import value_is_target_in_texts
    t1 = "設定董事及員工年度誠信經營教育訓練完訓率為100%，深化誠信文化。"
    t2 = "合計有7名董事及688名員工完成訓練，完訓率為100%。轄下子公司完訓率皆為100%。"
    assert value_is_target_in_texts("誠信經營教育訓練完訓率", "100 %", [t1, t2]) is False


def test_rule_based_extraction_skips_targets(monkeypatch):
    """真實片段（p47）：2035 年的長期減碳「目標」曾被當成 KPI 抽出（累計減碳＝63%）。"""
    import interpretation_agent as ia
    txt = ("依循金控 SBT 減碳目標，以2021 年為基準年，訂定，\n"
           "訂定短期（2025 年）累計減碳18%、中期（2030 年）累計減碳40.5%、長期（2035 年）\n"
           "累計減碳63%。\n綠色營運\n2.1\n")
    real = ia.context_is_target
    monkeypatch.setattr(ia, "context_is_target", lambda *a, **k: False)
    assert any(k.value == "63" for k in ia.extract_kpis(txt, "E"))      # 前提：沒有過濾時確實會被抽出
    monkeypatch.setattr(ia, "context_is_target", real)
    assert not any(k.value in ("18", "40.5", "63") for k in ia.extract_kpis(txt, "E"))


def test_actual_cues_override_target_words():
    """句子裡同時有「目標」與「達成、實際」→ 是結果，不是承諾。"""
    from citation import context_is_target
    s = "本公司已達成減碳目標，2024年實際減少 43,319 噸"
    assert not context_is_target(s, s.index("43"))


def test_kpi_name_with_corrupted_year_is_repaired():
    import interpretation_agent as ia
    assert ia._clean_kpi_name("投資組合財務碳排放量 (202決算)") == "投資組合財務碳排放量"
    assert ia._clean_kpi_name("財務碳排放強度 (2024)") == "財務碳排放強度 (2024)"


def test_pipeline_drops_target_kpi_but_keeps_actual(monkeypatch, patch_embedding):
    import interpretation_agent as ia
    from conftest import make_section
    secs = [make_section(section_id="r::1", company="示範公司", pillar="G",
                         text="設定董事及員工年度誠信經營教育訓練完訓率為100%，深化誠信文化。"),
            make_section(section_id="r::2", company="示範公司", pillar="G",
                         text="本公司合計有7名董事及688名員工完成訓練，資安事件通報共計35件。")]
    answer = ("【亮點 Highlights】\n- 這是一段足夠長度的亮點說明文字內容\n"
              "【待改善 Weaknesses】\n- 這是一段足夠長度的待改善說明文字內容\n"
              "【關鍵績效指標 KPIs】\n- 誠信經營教育訓練完訓率：100 %\n- 資安事件通報件數：35 件\n- 完成訓練員工人數：688 人\n"
              "【面向總結】\n整體表現尚可。")
    monkeypatch.setattr(ia, "call_llm", lambda *a, **k: answer)
    pa = ia.analyze_pillar("G", secs, "示範公司", "2024")
    names = [k.name for k in pa.kpis]
    assert not any("完訓率" in n for n in names)          # 只有「設定…」的語境 → 是目標
    assert any("資安事件通報件數" in n for n in names)
    assert any("完成訓練員工人數" in n for n in names)


# ── 個位數的 KPI 值 ────────────────────────────────────────────────
def test_single_digit_value_is_checked_when_it_is_the_only_number():
    from citation import primary_token
    assert primary_token("1 (無量尺標準)") == "1"
    assert primary_token("3 小時/場") == "3"
    assert primary_token("1/7 (約 14.3 %)") == "14.3"      # 有 2 位數以上的數字時優先使用它
    assert primary_token("2024 年") is None                  # 只有年份：不用來比對
    assert primary_token("3 家 4 次") is None                # 有多個個位數：無法決定用哪一個，不下結論
    near = "資料品質分數\n1\n1\n1.23\n金控管理目標\n"
    far = "董事兼職上限：董事不宜同時擔任超過四家上市上櫃公司之董事，並每年定期評估。"
    assert value_near_name_in_texts("資料品質分數", "1 (無量尺標準)", [near]) is True
    assert value_near_name_in_texts("資料品質分數", "1 (無量尺標準)", [far]) is not True


def test_single_digit_kpi_is_cited_to_the_chunk_that_has_its_name(patch_embedding):
    """真實案例：資料品質分數＝1 原本被引用到 p76，實際在 p58 的表格。"""
    from citation import select_sources_for_value
    from conftest import make_section
    table = make_section(section_id="r::58", page_range="p58",
                         text="盤查覆蓋率\n38%\n49%\n57%\n資料品質分數\n1\n1\n1.23\n金控管理目標\n")
    other = make_section(section_id="r::76", page_range="p76",
                         text="董事兼職上限：每年定期評估董事長之兼職情形，第1次會議提出報告。")
    cites = select_sources_for_value("資料品質分數: 1", "1", [other, table], "示範公司", "2024")
    assert cites[0].chunk_id == "r::58"


# ── 目標過濾的誤判回歸：實績句裡剛好有「制定」 ─────────────────────
def test_actual_count_in_a_sentence_mentioning_a_policy_is_not_a_target():
    """真實案例（p37）：「參考國際申訴處理機制規範制定《客戶申訴處理作業辦法》，2024 年立案之客訴案件共56 件」。
    56 是實績，曾因為「制定」被誤判成目標而遭捨棄。"""
    from citation import context_is_target, value_is_target_in_texts
    t = ("本公司參考國際申訴處理機制規範制定《客戶申訴處理作業辦法》，"
         "2024 年立案之客訴案件共56 件，同年皆已完成結案，結案率100%。")
    assert not context_is_target(t, t.index("56"))
    assert value_is_target_in_texts("客訴案件立案件數", "56 件", [t]) is False


# ── KPI 區塊標題的各種寫法 ─────────────────────────────────────────
def test_kpi_block_is_found_under_alternative_headers(monkeypatch, patch_embedding):
    import interpretation_agent as ia
    from conftest import make_section
    secs = [make_section(section_id="r::1", company="示範公司", pillar="S",
                         text="2024年資安教育訓練人數 723 人；證券商業務員證照取得 526 人次；期貨商業務員證照取得 386 人次。")]
    answer = ("【亮點 Highlights】\n- 這是一段足夠長度的亮點說明文字內容\n"
              "【待改善 Weaknesses】\n- 這是一段足夠長度的待改善說明文字內容\n"
              "### 【量化指標】\n- 資安教育訓練人數：723 人\n- 證券商業務員證照取得：526 人次\n- 期貨商業務員證照取得：386 人次\n"
              "### 【面向總結】\n整體表現尚可。")
    calls = []
    monkeypatch.setattr(ia, "call_llm", lambda *a, **k: (calls.append(1), answer)[1])
    pa = ia.analyze_pillar("S", secs, "示範公司", "2024")
    assert len(calls) == 1                       # 不需要額外補問
    assert len(pa.kpis) >= 3


def test_policy_formulation_wording_is_not_a_target_cue():
    """「制定」多半是『制定了某辦法』的過去事實，不是承諾的目標。"""
    from citation import context_is_target
    s = "2024年制定客訴處理辦法並處理 56 件"
    assert not context_is_target(s, s.index("56"))


def test_count_right_after_gong_is_actual_even_with_target_words_nearby():
    """緊鄰數值的「共」（共56件）代表已發生的合計，優先於前面出現的目標字眼。"""
    from citation import context_is_target
    s = "本年度設定改善目標，客訴案件共56件"
    assert not context_is_target(s, s.index("56"))


# ── 個位數 KPI 值：只採用名稱就在數字旁邊的片段（真實案例：資料品質分數＝1）──────────
def test_single_digit_kpi_ignores_chunks_where_the_digit_is_far_from_the_name(patch_embedding):
    from citation import select_sources_for_value
    from conftest import make_section
    # 這個片段與 KPI 名稱的文字非常相似（相似度高），但 1 離名稱很遠，只是剛好有個「第1次」
    similar = make_section(section_id="r::26", page_range="p26-27",
                           text=("資料品質分數的計算方式與說明，資料品質分數的評分標準如下。" * 4
                                 + "其他說明文字。" * 12 + "本年度會議第1次召開。"))
    table = make_section(section_id="r::58", page_range="p58",
                         text="盤查覆蓋率\n38%\n資料品質分數\n1\n1\n1.23\n")
    cites = select_sources_for_value("2024 年資料品質分數: 1", "1", [similar, table],
                                     "示範公司", "2024")
    assert [c.chunk_id for c in cites] == ["r::58"]


def test_speculative_remarks_are_removed_from_kpi_values():
    import interpretation_agent as ia
    assert ia._clean_kpi_value("1 分（滿分可能為較高分值，需對照內部標準）") == "1 分"
    assert ia._clean_kpi_value("100 %（確保專責單位執行力）") == "100 %（確保專責單位執行力）"   # 非揣測的補充保留
    assert ia._clean_kpi_value("43,319 tCO2e") == "43,319 tCO2e"


def test_vague_kpi_names_are_rejected():
    import interpretation_agent as ia
    assert not ia._valid_kpi("調查結果", "8 件")
    assert ia._valid_kpi("不受理之檢舉案件數", "8 件")
