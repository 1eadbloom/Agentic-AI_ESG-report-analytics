"""QC 與重試路由的回歸測試（對應 trace 中發現的漏洞：總數夠就放行，G 面向 0 條也通過）。"""
import coordinator_agent as ca


def _state(**kw):
    base = dict(file_name="x.pdf", company="示範公司", model="m", rebuild=False,
                sections=[], diagnosis={}, solution={}, retry_count=0,
                qc_passed=False, retry_target="interpret", errors=[], report_path="")
    base.update(kw)
    return base


def _good_diag():
    item = {"text": "這是一段足夠長度的分析說明"}
    return {p: {"highlights": [item], "weaknesses": [item]} for p in "ESG"}


def _strategies(**counts):
    return {"strategies": [{"pillar": p, "term": "short"}
                           for p, n in counts.items() for _ in range(n)]}


def test_pillar_with_no_strategies_fails_and_retries_only_solution():
    r = ca.node_quality_check(_state(diagnosis=_good_diag(),
                                     solution=_strategies(E=5, S=4, G=0)))
    assert r["qc_passed"] is False
    assert r["retry_target"] == "solution"
    assert any("G 面向策略不足" in e for e in r["errors"])


def test_placeholder_diagnosis_fails_and_retries_interpretation():
    bad = {p: {"highlights": [{"text": "（本份報告書中未找到相關內容）"}],
               "weaknesses": [{"text": "###"}]} for p in "ESG"}
    r = ca.node_quality_check(_state(diagnosis=bad,
                                     solution=_strategies(E=5, S=5, G=5)))
    assert r["qc_passed"] is False and r["retry_target"] == "interpret"


def test_healthy_run_passes():
    r = ca.node_quality_check(_state(diagnosis=_good_diag(),
                                     solution=_strategies(E=5, S=5, G=5)))
    assert r["qc_passed"] is True


def test_search_failure_is_not_retried():
    s = _state(qc_passed=False, retry_count=1, retry_target="interpret",
               errors=["找不到檔案：C:\\x.pdf"])
    assert ca.route_after_qc(s) == "report"


def test_retry_target_routes_back_to_the_right_node():
    assert ca.route_after_qc(_state(qc_passed=False, retry_count=1,
                                    retry_target="solution")) == "solution"
    assert ca.route_after_qc(_state(qc_passed=False, retry_count=1,
                                    retry_target="interpret")) == "interpret"
    assert ca.route_after_qc(_state(qc_passed=False, retry_count=2)) == "report"


def test_solution_is_skipped_when_interpretation_produced_nothing(monkeypatch):
    """Interpretation 整個失敗（例如 LLM 服務崩潰）時，不該再用空診斷呼叫 LLM 產生策略。"""
    def boom(*a, **k):
        raise AssertionError("不應該呼叫 run_solution_agent")
    monkeypatch.setattr(ca, "run_solution_agent", boom)
    r = ca.node_solution(_state(diagnosis={}, errors=["Interpretation Agent 失敗：連線中斷"]))
    assert r["solution"] == {}
    assert r["errors"] == ["Interpretation Agent 失敗：連線中斷"]


def test_failed_interpretation_leads_to_interpret_retry():
    """跳過策略生成後，QC 要把問題判成「診斷缺漏」，並回到 interpret（而不是只重跑 solution）。"""
    r = ca.node_quality_check(_state(diagnosis={}, solution={}))
    assert r["qc_passed"] is False and r["retry_target"] == "interpret"
