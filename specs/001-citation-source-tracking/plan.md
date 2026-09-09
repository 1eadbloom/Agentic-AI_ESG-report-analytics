# Implementation Plan: Citation Source Tracking

**Branch**: `001-citation-source-tracking` | **Date**: 2026-09-08 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-citation-source-tracking/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command; its definition describes the execution workflow.

## Summary

Add source-citation tracking to the ESG report analysis output. Every conclusion item — highlights, weaknesses, KPIs, and strategies — must carry references to the original document fragments that informed it (source company, report year, chunk ID, page range, and a brief excerpt), so users can trace and verify AI output. The citation mechanism must preserve the existing three-layer metadata isolation (cross-company, cross-year, report vs. knowledge base) and must not change the analysis behavior itself.

Technical approach: introduce a `Citation` entity derived from existing section metadata; deterministically attribute each conclusion to source fragments using embedding-similarity scoring (with keyword-overlap fallback for rule-extracted KPIs) — NOT LLM self-reporting, per spec Assumption. Propagate citations through the Interpretation and Solution agents and render them in both Markdown and JSON final outputs.

## Technical Context

**Language/Version**: Python >= 3.10

**Primary Dependencies**: langgraph, chromadb, sentence-transformers, litellm, pdfplumber, pypdf, python-dotenv

**Storage**: ChromaDB vector store (`esg_reports` collection); local files — `data/processed/*.json` (parsed sections), `outputs/*.md` + `outputs/*.json` (final reports)

**Testing**: pytest (to be added; no test framework currently present — see research.md R5)

**Target Platform**: Local CLI (Python venv, cross-platform; dev machine is Windows)

**Project Type**: CLI tool + multi-agent pipeline (module-style `.py` files at repo root with CLI entry points)

**Performance Goals**: SC-004 — report generation time must not increase by more than 10%; attribution must be computationally cheap relative to LLM calls (embedding scoring reuses the already-loaded model; no extra LLM round-trips)

**Constraints**: Three-layer metadata isolation (cross-company, cross-year, report vs. knowledge_base) must be preserved; Markdown output must remain readable (SC-005); analysis behavior unchanged (FR-009)

**Scale/Scope**: Small scale — a handful of reports/companies, tens of chunks per report, ~15 attributed items per report (3–5 highlights + 2–4 weaknesses + ≤10 KPIs per pillar + ≤15 strategies)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution (`.specify/memory/constitution.md`) is an **uninitialized template** — all principle/gate sections contain only placeholder values (`[PRINCIPLE_1_NAME]`, etc.). No binding gates or governing constraints are defined.

**Gate result: PASS (no constraints defined).** Proceeding with Phase 0/1 under best-practice defaults; re-checked after Phase 1 design below.

## Project Structure

### Documentation (this feature)

```text
specs/001-citation-source-tracking/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   ├── output-json.md   # Contract: structured JSON output schema
│   └── markdown-report.md # Contract: Markdown citation rendering rules
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Single project (flat module layout — existing structure, no restructuring)
esg_utils.py              # shared paths, LLM call, embedding fn, ChromaDB helpers
search_agent.py           # PDF parsing → ESGSection list (existing; unchanged)
interpretation_agent.py   # per-pillar analysis (extend: citations on items)
solution_agent.py         # strategies + best practices (extend: citations on strategies)
coordinator_agent.py      # LangGraph orchestration + report rendering (extend: citations in md/json)
citation.py               # NEW: Citation entity + deterministic attribution logic
tests/
├── unit/                 # NEW: test_citation.py, test_attribution.py, test_rendering.py
├── integration/          # NEW: test_coordinator_citations.py
└── fixtures/             # NEW: tiny .txt report fixture(s)
```

**Structure Decision**: The repository uses flat root-level Python modules with CLI entry points (no `src/` package). This feature follows that convention: citation logic lives in a new self-contained module (`citation.py`) so it is independently unit-testable without LLM/model dependencies, while the existing agents are minimally extended to pass section metadata along. Tests are added under a new `tests/` tree (currently absent).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations — the constitution template defines no gates.

**Post-Phase-1 Constitution Check (re-checked after design): PASS** — no constitution constraints were violated or need justification. Design introduces one new module and field-level extensions only.