# Research: Citation Source Tracking

**Date**: 2026-09-08
**Scope**: Design decisions for `specs/001-citation-source-tracking`
**Source**: Codebase analysis of `search_agent.py`, `interpretation_agent.py`, `solution_agent.py`, `coordinator_agent.py`, `esg_utils.py`, plus the feature spec.

---

## R1 — How to attribute conclusions to source fragments

**Decision**: Deterministic, non-LLM attribution. Score each generated conclusion against the candidate sections it was derived from using **embedding cosine similarity** (reusing the already-loaded embedding model via the existing singleton), select the **top-3 sections above a threshold** as citations. For rule-extracted items (KPIs via regex in `interpretation_agent.extract_kpis`), attribute directly and exactly to the section where the regex matched — no similarity scoring needed.

**Rationale**:
- The spec Assumption explicitly rules out LLM self-reporting ("citations are generated deterministically… not by asking the LLM to self-report its sources").
- The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) is already loaded as a process-wide singleton in `esg_utils.get_embedding_fn()` — scoring a handful of conclusions (≈15/report) against ≤15 candidate sections adds only milliseconds, satisfying SC-004.
- Embeddings tolerate LLM paraphrase better than keyword matching (a highlight rarely contains the exact source words).
- KPI regex extraction has an exact provenance — the match object knows which section it came from.

**Alternative considered**:
- Keyword/term-overlap scoring: free, but fragile for paraphrased prose; used only as a no-embedding fallback.
- LLM self-report sources: explicitly rejected by spec assumption (unreliable/hallucinated citations).

**Threshold rule** (unattributed handling, per spec FR-006): if no section scores above the minimum similarity threshold (default 0.30, configurable), the conclusion is labeled `unattributed` with an explanatory note instead of guessing.

---

## R2 — Data representation of attributed conclusions

**Decision**: Add a `Citation` dataclass. Change `PillarAnalysis.highlights` / `.weaknesses` from `list[str]` to `list[AnalysisItem]` where `AnalysisItem = {text, citations}`. Add a `citations` field to `KPIItem` and `StrategyItem` (default empty). `DiagnosisResult`, `SolutionResult` and `ESGSection` otherwise unchanged.

**Rationale**:
- Keeps conclusions and their evidence co-located in the data model, naturally serializing into both Markdown and JSON.
- Field-level extension keeps the existing pipeline's analytical outputs byte-identical (FR-009) — the LLM prompt/take-away text does not change; only the wrapper/storage shape does.
- `KPIItem`, `StrategyItem`, `PillarAnalysis` are already dataclasses with `asdict()` serialization; extending fields is low-risk.

**Alternative considered**: Parallel index-mapped arrays (`highlights=[...]`, `highlight_sources=[[...]]`) — rejected: error-prone to keep in sync, awkward to serialize, worse for programmatic consumers.

---

## R3 — Where in the pipeline citations are created

**Decision**:
1. **Interpretation Agent**: `analyze_pillar()` is passed the pillar's **section list** (not just the concatenated text). After LLM output parsing, each highlight/weakness and each KPI is attributed against those sections (R1). Citations carry `company`, `report_year`, `source_file`, `chunk_id` (= `section_id`), `page_range`, `doc_type` and an `excerpt` (first ~100 chars of section text).
2. **Solution Agent**: each strategy is generated from a pillar's weaknesses, so a strategy **inherits the union of its pillar's weakness citations**. GRI gaps are rule-based → no citations. Best practices keep the existing `source` reference (already present), optionally extended with an excerpt.
3. **Coordinator Agent**: report rendering consumes the citations and formats them per the contracts (R4). No change to the LangGraph topology.

**Isolation enforcement (three-layer metadata)**:
- Interpretation never sees sections outside the current run's scoped set (`state["sections"]` filtered by pillar for the running company/year/file).
- ChromaDB-backed retrieval (the fallback path in `run_interpretation_agent`) already filters by `source_file` + `doc_type="report"`; citations are built only from the filtered results.
- A guard in the attribution function asserts every cited section's `company == target company`, `report_year == target year`, and records `doc_type` truthfully (report vs. knowledge_base), never mixing them.

**Rationale**: Traceability is anchored at the exact point where analysis text is consumed (interpret; strategy), so provenance never has to be reverse-engineered later.

---

## R4 — Output contracts (Markdown + JSON)

**Decision**:
- **Markdown**: compact inline parenthetical citation after each item:
  `(來源: {公司} · {年度} · chunk {chunk_id} · {page_range})`, multiple sources joined with `; `. Unattributed: `(非文件依據 · 通用知識/經驗判斷)`.
- **JSON**: each attributed item gains a `sources` array of objects:
  `{"company", "year", "source_file", "chunk_id", "page_range", "doc_type", "excerpt"}`.
- Full field-level schemas are captured in `contracts/output-json.md` and `contracts/markdown-report.md`.

**Rationale**: Inline Markdown keeps the report readable (SC-005) and self-contained; structured JSON is machine-verifiable (SC-001/SC-002).

**Alternative considered**: Footnote-style `[1]` references with a bibliography section — rejected: breaks reading flow for a report whose value is verifiability-in-place.

---

## R5 — Testing strategy

**Decision**: Add `pytest`. Test the citation logic as **pure functions with no LLM/model dependency**:
- Unit: `Citation`/`AnalysisItem` serialization; attribution scoring & thresholds; unattributed labeling; keyword-exact KPI provenance.
- Isolation: synthetic sections from two companies/two years across report & knowledge_base doc types → assert zero cross-contamination (SC-002).
- Rendering: Markdown + JSON golden-output checks.
- Integration: run `coordinator_agent.py` against a small `.txt` fixture (no PDF/big-embedding download needed for the inference path) with mocked `call_llm`, and assert every non-unattributed item has valid citations and JSON conforms to the contract.

**Rationale**: The highest-value tests are pure-logic, offline, and fast. Mocking `call_llm` and the embedding fetcher keeps CI-free validation possible on any machine. No LLM/network time in the unit layer keeps the suite under a few seconds (SC-004 spirit: cheap attribution).

**Alternative considered**: stdlib `unittest` to avoid a new dependency — rejected: pytest's fixtures and `assert` introspection make the golden/negative test matrix much more readable; adding it to `requirements.txt` is a one-line change.

---

## Dependency/Integration notes

- **New module** `citation.py` depends only on `esg_utils` (`get_embedding_fn`) — decoupled from LangGraph so it is unit-testable standalone.
- **Backward compatibility**: existing `outputs/*.json` files have no `sources` field; the new schema is additive (old consumers reading only `company`/`diagnosis`/`solution` fields keep working).
- **No changes needed** to `search_agent.py` (sections already carry all required metadata) or to the LangGraph topology in `coordinator_agent.py`.