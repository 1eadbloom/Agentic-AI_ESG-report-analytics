---

description: "Task list for citation source tracking feature implementation"
---

# Tasks: Citation Source Tracking

**Input**: Design documents from `/specs/001-citation-source-tracking/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests ARE included — the feature's validation approach is test-first (see `quickstart.md` and research.md R5): unit & isolation tests validate the pure attribution logic offline; E2E test validates contract conformance with a mocked LLM.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: modules at repository root (`esg_utils.py`, `coordinator_agent.py`, etc.), `tests/` at repository root (per plan.md Structure Decision)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Add `pytest` to `requirements.txt` (dev dependency) and create `pytest.ini` with `testpaths = tests`
- [X] T002 [P] Create tests directory tree: `tests/unit/`, `tests/integration/`, `tests/fixtures/`
- [X] T003 [P] Create minimal E/S/G report fixture `tests/fixtures/example_report.txt` with clearly separated Environment / Social / Governance paragraphs (used by attribution and E2E tests; company `示範公司`, year 2024)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core citation module — MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Create `citation.py` with `Citation` dataclass (fields per data-model.md: company, report_year, source_file, chunk_id, page_range, doc_type, excerpt) and `AnalysisItem` dataclass (text, citations)
- [X] T005 [P] Implement attribution scoring in `citation.py`: `score_against_sections(text, sections)` using cosine similarity via `esg_utils.get_embedding_fn()`, with keyword-overlap fallback
- [X] T006 [P] Implement source selection in `citation.py`: `select_sources(text, sections, top_k=3, threshold=0.30)` per research.md R1
- [X] T007 Implement isolation guard in `citation.py`: `assert_section_scope(section, company, report_year)` — raises on company/`report_year`/`doc_type` mismatch (FR-004)
- [X] T008 [P] Implement `create_unattributed_citation(run_company, run_year)` in `citation.py` per FR-006 (chunk_id `"unattributed"`, empty source_file)

**Checkpoint**: Foundation ready — `citation.py` exposes data structures + attribution + isolation guard; user story implementation can begin

---

## Phase 3: User Story 1 - Verify a Specific Conclusion's Source (Priority: P1) 🎯 MVP

**Goal**: Every highlight, weakness, KPI, and strategy item carries citations (company, year, chunk ID, page range); multi-source conclusions list all contributors; Markdown + JSON outputs both include the citations.

**Independent Test**: Run `pytest tests/unit/test_rendering.py tests/integration/test_coordinator_citations.py` + generate a report for the fixture and verify every non-unattributed highlight/weakness/KPI/strategy has a valid citation in both `outputs/*.md` and `outputs/*.json`.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T009 [P] [US1] Unit test `Citation`/`AnalysisItem` field construction + serialization in `tests/unit/test_citation.py`
- [X] T010 [P] [US1] Unit test attribution scoring & top-k threshold selection (incl. unattributed below 0.30) in `tests/unit/test_attribution.py`
- [X] T011 [P] [US1] Unit test Markdown rendering per `contracts/markdown-report.md` in `tests/unit/test_rendering.py`
- [X] T012 [US1] Unit test JSON schema conformance per `contracts/output-json.md` (additive fields, `sources[]` arrays) in `tests/unit/test_output_json_contract.py`

### Implementation for User Story 1

- [X] T013 [P] [US1] Extend `KPIItem` with `citations: list = field(default_factory=list)` in `interpretation_agent.py` (default `[]`, backward compatible)
- [X] T014 [P] [US1] Change `PillarAnalysis.highlights` / `.weaknesses` to `list[AnalysisItem]` in `interpretation_agent.py` (constructed from plain strings in parsing)
- [X] T015 [US1] Modify `analyze_pillar()` in `interpretation_agent.py` to accept the pillar's `ESGSection` list (not just concatenated text) (depends on T013, T014)
- [X] T016 [US1] Attribute highlights/weaknesses to sections via `select_sources` and KPIs to their exact regex-matched section inside `analyze_pillar()` (depends on T015)
- [X] T017 [US1] Extend `StrategyItem` with `citations` field in `solution_agent.py`
- [X] T018 [US1] Implement strategy citation inheritance (union of its pillar's weakness citations per data-model.md) in `generate_strategies()` in `solution_agent.py` (depends on T017)
- [X] T019 [US1] Update `node_interpret` in `coordinator_agent.py` to pass per-pillar section lists to `run_interpretation_agent` (depends on T015)
- [X] T020 [US1] Render inline citations in Markdown report in `node_report` in `coordinator_agent.py` per `contracts/markdown-report.md` (depends on T016, T018)
- [X] T021 [US1] Render `sources` arrays in JSON output in `node_report` in `coordinator_agent.py` per `contracts/output-json.md` (depends on T016, T018)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently (MVP)

---

## Phase 4: User Story 2 - Cross-Reference Citations Across Companies (Priority: P2)

**Goal**: Zero cross-company / cross-year / report-KB contamination — citations never reference fragments outside the run's scope (SC-002).

**Independent Test**: Run `pytest tests/unit/test_isolation_guard.py tests/integration/test_isolation.py` — synthetic sections for (CompanyA,2024), (CompanyB,2023), knowledge-base docs; assert no Citation for CompanyA references CompanyB or year-2023 fragments, and KB sources are never mixed with report sources in one conclusion.

### Tests for User Story 2 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T022 [P] [US2] Unit test `assert_section_scope` + `select_sources` isolation rejection cases in `tests/unit/test_isolation_guard.py`
- [X] T023 [US2] Integration test with mixed synthetic sections (two companies, two years, KB) driving `analyze_pillar` in `tests/integration/test_isolation.py` — asserts zero contamination in output citations

### Implementation for User Story 2

- [X] T024 [US2] Wire `assert_section_scope` into the attribution call path in `analyze_pillar()` in `interpretation_agent.py` (depends on T023)
- [X] T025 [US2] Strengthen ChromaDB retrieval filter in `run_interpretation_agent` (interpretation_agent.py) to scope by `company` AND `report_year` (in addition to existing `source_file` + `doc_type`) when sections are not passed in

**Checkpoint**: User Stories 1 AND 2 both work independently

---

## Phase 5: User Story 3 - Trace a Citation Back to Original Text (Priority: P3)

**Goal**: Citations include a brief excerpt of the original source text (≤100 chars, FR-008), displayed with the citation in Markdown and stored in JSON.

**Independent Test**: Run `pytest tests/unit/test_excerpt.py tests/unit/test_rendering.py` and inspect a generated report — each non-KB source shows a quoted excerpt ≤100 chars; markdown display truncated at 60 chars with `…`.

### Tests for User Story 3 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T026 [P] [US3] Unit test excerpt truncation (≤100 stored, ≤60 displayed with ellipsis) in `tests/unit/test_excerpt.py`

### Implementation for User Story 3

- [X] T027 [US3] Populate `excerpt` when constructing `Citation` from a section in `citation.py` (leading text, truncated to 100 chars)
- [X] T028 [US3] Render displayed excerpt (60-char cap) in Markdown citation line in `node_report` in `coordinator_agent.py` per `contracts/markdown-report.md` (depends on T027)
- [X] T029 [US3] Confirm `excerpt` included in JSON `sources` objects (extend JSON contract test from T012 to assert non-empty excerpt)

**Checkpoint**: All user stories independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T030 [P] Update README.md documenting the citation feature, output formats, and reference to `specs/001-citation-source-tracking/`
- [X] T031 Run full validation per `quickstart.md` (unit + isolation + E2E suites) and verify all pass
- [X] T032 Run performance sanity check per quickstart.md §5 — report generation time increase ≤10% (SC-004)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational (Phase 2) completion
  - Stories proceed sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Depends only on Phase 2 — no dependency on other stories (MVP)
- **User Story 2 (P2)**: Depends on Phase 2 (isolation guard T007) + US1 wiring (T015/T016); independently testable via T022/T023
- **User Story 3 (P3)**: Depends on Phase 2 (Citation construction) + US1 rendering (T020/T021); independently testable via T026

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Data structures before attribution logic before rendering
- US1 progression: entities (T013/T014/T017) → attribution wiring (T015/T016/T018) → coordinator render (T019/T020/T021)

### Parallel Opportunities

- All Phase 1 tasks marked [P] can run in parallel
- Phase 2: T005, T006, T008 are independent (watch: T004 must land first — it is not marked [P])
- US1 tests T009–T012 all [P]; entity edits T013/T014/T017 all [P]
- US2: T022 [P] independent of implementation; T023 drives T024's wiring
- US3: T027 runs after T026 fail, before T028/T029
- T030 [P] in polish runs in parallel with T031/T032

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together (written first, run red):
Task: "Unit test Citation/AnalysisItem serialization in tests/unit/test_citation.py"
Task: "Unit test attribution scoring & threshold in tests/unit/test_attribution.py"
Task: "Unit test Markdown rendering in tests/unit/test_rendering.py"
Task: "Unit test JSON contract conformance in tests/unit/test_output_json_contract.py"

# Launch independent entity edits together:
Task: "Extend KPIItem with citations in interpretation_agent.py"
Task: "Change PillarAnalysis highlights/weaknesses to AnalysisItem in interpretation_agent.py"
Task: "Extend StrategyItem with citations in solution_agent.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (`citation.py`)
3. Complete Phase 3: User Story 1 (citations on all conclusions + render markdown/JSON)
4. **STOP and VALIDATE**: Run US1 independent test; verify citations appear per contracts
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → citation core ready
2. Add US1 → citations appear everywhere → validate → **MVP**
3. Add US2 → isolation guarantee + strengthened retrieval filter → validate → demo
4. Add US3 → excerpts make citations self-verifiable → validate → demo
5. Each story adds value without breaking previous stories (FR-009: analysis text unchanged)

### Parallel Team Strategy

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: waits for US1 entities, then User Story 2
   - Developer C: waits for US1 rendering, then User Story 3
3. Polish phase is a single-pass sweep by one owner

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail (TDD) before implementing each story
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- `citation.py` must stay free of LangGraph/LLM imports to remain independently unit-testable (research.md R5)
- Avoid: vague tasks, same-file parallel conflicts, cross-story dependencies that break independence (T015/T016 share `analyze_pillar` — do not run in parallel)

---

## Phase 7: Convergence

**Purpose**: Close gaps found by `/speckit.converge` against the post-implement codebase. Tasks archive the intent from `spec.md` / `plan.md` / `tasks.md`.

- [X] T033 [US2] Preserve `section_id` into citations from the ChromaDB retrieval path in `run_interpretation_agent` (interpretation_agent.py, T025 no-sections branch) — thread `res["ids"][0]` into `_meta_section` so every non-unattributed Citation has non-empty `chunk_id`, and add a unit test covering the no-sections retrieval branch with a fake collection; also fix `search_agent.py` `process_file` to stop returning `[]` silently when the file is unchanged (or otherwise ensure `node_interpret` never falls through to the retrieval path with missing chunk IDs) per FR-002 (partial)
- [X] T034 Implement a repeatable end-to-end performance baseline/guard for SC-004 per quickstart.md §5 — record T0 (pre-feature) and T1 (post-feature) wall-clock times on the mocked-LLM path and assert `T1 ≤ 1.10 · T0`, or backfill the measured T0/T1 values into quickstart.md §5 (partial)

---

## Phase 8: Convergence

**Purpose**: Close gaps found by the second `/speckit.converge` pass against the post-implement codebase. Tasks archive the intent from `spec.md` / `plan.md` / `tasks.md`.

- [X] T035 Align the test-name mapping table in quickstart.md §1 with the actual test functions per quickstart.md §1 / US3 — replace the stale names `test_excerpt_max_100` → `test_truncate_max_100` (tests/unit/test_excerpt.py) and `test_render_markdown`/`test_render_json` → `test_render_single_source`/`test_render_multiple_sources`/`test_render_unattributed`/`test_render_knowledge_base` (tests/unit/test_rendering.py) so the FR↔test traceability claims hold (partial)