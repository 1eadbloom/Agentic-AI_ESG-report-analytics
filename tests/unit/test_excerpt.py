"""T026 + FR-008 驗證：摘錄長度限制（儲存 ≤100、顯示 ≤60）。"""
from citation import display_excerpt, truncate
from conftest import make_section

SEC_TEXT = ("本公司已完成 2024 年度溫室氣體盤查作業，並透過第三方機構取得 "
            "ISO 14064 確信，整體碳排放量較前一年度下降約 12%。再生能源使用比例 "
            "逐年提升，綠電採購量持續增加，並導入 ISO 50001 能源管理系統。"
            "水資源管理方面，全廠區執行取水與排放監測，並定期公布節水目標。"
            "廢棄物部分則以 3R 原則推動資源循環，並逐年降低最終處置量。"
            "2024 年度溫室氣體排放量為 12000 公噸，再生能源使用比例為 15%。")


def test_truncate_max_100():
    assert len(truncate(SEC_TEXT)) <= 100
    assert len(truncate("短")) == 1


def test_display_excerpt_cap_60():
    short = "乾淨的emissions數據"
    assert display_excerpt(short) == short
    long_t = "甲" * 120
    d = display_excerpt(long_t)
    assert len(d.partition("…")[0]) == 60
    assert d.endswith("…")


def test_select_sources_excerpt_from_section(patch_embedding):
    """Citation 的 excerpt 必須來自來源片段文字且 ≤100。"""
    from citation import select_sources
    sec = make_section(section_id="r.pdf::2", pillar="E", text=SEC_TEXT)
    text = "本公司已完成 2024 年度溫室氣體盤查作業並取得 ISO 14064 確信。"
    cites = select_sources(text, [sec], "示範公司", "2024")
    assert len(cites) == 1
    assert len(cites[0].excerpt) <= 100
    assert cites[0].excerpt in SEC_TEXT  # 確為片段前綴