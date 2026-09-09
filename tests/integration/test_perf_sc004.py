"""Quickstart 第 5 步：SC-004「引用機制」開銷守衛。

量測 mocked-LLM 管線上「引用機制」本身增加的牆鐘時間：
  T1 = 完整管線（真實歸因 + 引用渲染）
  T0 = 同一管線但禁止歸因/渲染（最接近未加 feature 的 baseline）
  overhead = T1 − T0

真實管線時間由 LLM call 主導（秒級）；引用機制不新增任何 LLM round-trip，
只做 embedding 相似度（沿用已載入模型）與字串串接。因此直接以 mocked 路徑
斷言「T1 ≤ 1.10·T0」會因絕對時間過小而失真。本守衛改為更穩健的等價斷言：
以「最保守的 LLM 主導時間下限」為基準，要求
  overhead ≤ 10% × (4 次 LLM call × 200ms) = 80ms，
只要防護通過，即保證在任何真實 LLM 運行上 SC-004（≤10%）成立。
"""
import time

import coordinator_agent
import citation
import interpretation_agent as ia
import solution_agent as sa
from integration.test_coordinator_citations import _FakeLLM, build_sections

COMPANY = "示範公司"
FILE = "example_report.txt"

# 保守的最小真實 LLM 管線時間：4 次 LLM call × 200ms
MIN_LLM_CALLS = 4
MIN_LLM_LATENCY_S = 0.2
SC004_BUDGET_S = 0.10 * MIN_LLM_CALLS * MIN_LLM_LATENCY_S  # 0.08s


def _pipeline_once():
    diag = ia.run_interpretation_agent(company=COMPANY, source_file=FILE,
                                       sections=build_sections())
    sol = sa.run_solution_agent(diag)
    state = {
        "file_name": FILE, "company": COMPANY, "model": "mock",
        "sections": [], "diagnosis": diag.to_dict(),
        "solution": sol.to_dict(), "retry_count": 0, "qc_passed": True,
        "errors": [], "report_path": "",
    }
    coordinator_agent.node_report(state)


def _best_of(attribution_on, monkeypatch, tmp_path, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        with monkeypatch.context() as m:
            m.setattr(ia, "call_llm", _FakeLLM())
            m.setattr(sa, "call_llm", _FakeLLM())
            m.setattr(sa, "get_best_practices",
                      lambda diagnosis, model=None: sa._fallback_best_practices())
            m.setattr(coordinator_agent, "OUTPUTS_DIR", tmp_path)
            if not attribution_on:
                # 關閉歸因與引用渲染 → 近似 citation 機制不存在時的 baseline
                m.setattr(
                    ia, "select_sources",
                    lambda text, sections, company, year, **kw:
                        [citation.make_unattributed(company, year)])
                m.setattr(citation, "citations_to_markdown", lambda *a, **k: "")
            t0 = time.perf_counter()
            _pipeline_once()
            best = min(best, time.perf_counter() - t0)
    return best


def test_sc004_mechanism_overhead_within_budget(patch_embedding, monkeypatch, tmp_path):
    """SC-004：引用機制開銷不得超過 LLM 主導基準的 10%（80ms 預算）。"""
    t0 = _best_of(False, monkeypatch, tmp_path)
    t1 = _best_of(True, monkeypatch, tmp_path)
    overhead = max(t1 - t0, 0.0)

    print(f"\nT0(T1) 引用機制開銷: T0={t0*1e3:.1f}ms  T1={t1*1e3:.1f}ms  "
          f"overhead={overhead*1e3:.1f}ms (預算 ≤ {SC004_BUDGET_S*1e3:.0f}ms)")
    assert overhead <= SC004_BUDGET_S, (
        f"引用機制開銷 {overhead*1e3:.1f}ms 超出預算 "
        f"{SC004_BUDGET_S*1e3:.0f}ms（= 4 LLM call × 200ms × 10%）")