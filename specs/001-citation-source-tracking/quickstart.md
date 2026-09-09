# Quickstart: Validating Citation Source Tracking

**Date**: 2026-09-08 | **Feature**: [spec.md](spec.md) | **Contracts**: [JSON](contracts/output-json.md), [Markdown](contracts/markdown-report.md) | **Data model**: [data-model.md](data-model.md)

This guide proves the feature works end-to-end. It is a validation/run guide only — full implementation detail lives in `tasks.md` (Phase 2).

## Prerequisites

- Python >= 3.10 venv with packages from `requirements.txt` (this feature adds `pytest`).
- A tiny local report fixture (no PDF needed): `tests/fixtures/example_report.txt` with a few E/S/G paragraphs that clearly differ per company/year so isolation is provable.

## 1. Unit tests — attribution & isolation logic (no LLM, no model download)

```bash
pytest tests/unit -v
```

**Expected**: all pass in seconds.

| Test | Proves |
|------|--------|
| `test_citation_fields` | FR-002 (citation carries company/year/chunk_id/page_range) |
| `test_attribution_topk` | FR-005 (multi-source citation lists all contributors) |
| `test_threshold_unattributed` | FR-006 (below-threshold → `unattributed` + note) |
| `test_kpi_exact_provenance` | KPI regex item cites exactly the matched section |
| `test_truncate_max_100` | FR-008 (excerpt ≤ 100 chars) |
| `test_render_single_source` / `test_render_multiple_sources` / `test_render_unattributed` / `test_render_knowledge_base` | Render rules in `contracts/markdown-report.md` / `contracts/output-json.md` |

## 2. Isolation tests — three-layer metadata (SC-002)

```bash
pytest tests/integration/test_isolation.py -v
```

Uses synthetic sections for (CompanyA, 2024), (CompanyB, 2023), and knowledge-base docs.

**Expected**: zero contamination — no citation for CompanyA references CompanyB; no year-2023 fragment cited for a 2024 run; KB sources always tagged `doc_type: knowledge_base` and never merged with report sources under one conclusion.

## 3. End-to-end CLI run (mocked LLM)

Run the coordinator on the fixture with a mocked `call_llm` (deterministic response) so the check is reproducible offline:

```bash
pytest tests/integration/test_coordinator_citations.py -v
# or manually:
python coordinator_agent.py --file example_report.txt --company "示範公司" --rebuild
```

**Expected outcomes**:
- Markdown report under `outputs/` shows inline citations per the [Markdown contract](contracts/markdown-report.md):
  - every Highlight / Weakness / KPI / strategy bullet ends with `(來源: …)` or the unattributed note
- JSON output under `outputs/` conforms to the [JSON contract](contracts/output-json.md):
  - every item in `highlights`, `weaknesses`, `kpis`, `strategies` has a `sources` array
  - every non-unattributed source has `company`, `report_year`, `source_file`, `chunk_id`, `page_range`; `excerpt` ≤ 100 chars
  - `company` of every source == `"示範公司"`, year == run year

## 4. Manual spot-check (real LLM, optional)

```bash
cp .env.example .env   # set LITELLM vars / Ollama
python coordinator_agent.py --file "2024年永續報告書_中_中信證券000616.pdf" --company "中信證券"
```

Open `outputs/中信證券_ESG分析報告_*.md` and verify: pick 3 conclusions, follow their chunk IDs to the corresponding pages — the source text should support the conclusion (user-story acceptance). Expected: identifying the source for any conclusion takes < 5 seconds (SC-003).

## 5. Performance sanity (SC-004)

The citation mechanism adds no LLM round-trips and reuses the already-loaded embedding
model (research R2/R4). Enforce the ≤10% bound with a repeatable guard that measures
only the mechanism's overhead on the mocked-LLM path:

```bash
pytest tests/integration/test_perf_sc004.py -v
```

**Methodology**: run the mocked pipeline twice, best-of-3 — `T1` = full pipeline with real
attribution + citation rendering; `T0` = same pipeline with attribution/rendering disabled
(the closest reproducible proxy for the pre-feature baseline). The guard asserts
`(T1 − T0) ≤ 0.10 · (4 LLM calls × 200ms) = 80ms`, a conservative lower bound for any real
LLM runtime; passing it therefore guarantees SC-004 (≤10%) on any real run.

**Recorded measurement (2026-09-08, dev machine, `test_perf_sc004.py`)**:

```text
T0 = 1.0 ms    T1 = 1.2 ms    overhead (T1 − T0) = 0.2 ms    budget ≤ 80 ms   PASS
```

> Note: a true pre-feature `T0` (before the citation feature existed) can no longer be
> measured on this codebase; the guard above is the enforceable, reproducible equivalent.

## Acceptance checklist mapping

| Quickstart step | Success criteria covered |
|-----------------|--------------------------|
| 1 unit tests | SC-001, SC-005 (via rendering goldens) |
| 2 isolation tests | SC-002 |
| 3 E2E CLI | SC-001, SC-003 (structure allows < 5s lookup), contract conformance |
| 4 manual spot-check | SC-003 (user experience) |
| 5 perf check | SC-004 |