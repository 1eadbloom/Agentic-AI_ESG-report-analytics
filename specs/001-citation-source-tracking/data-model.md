# Data Model: Citation Source Tracking

**Date**: 2026-09-08 | **Feature**: [spec.md](spec.md) | **Related**: [research.md](research.md)

This document defines the entities added/changed for citation tracking, their fields, validation rules, and how they flow through the pipeline.

---

## Entities

### 1. Citation (NEW)

A reference linking an AI-generated conclusion to the original source fragment it was derived from.

| Field | Type | Required | Description | Validation Rule |
|-------|------|----------|-------------|-----------------|
| `company` | string | yes | Source company name | Must equal the target company of the run (isolation layer 1) |
| `report_year` | string | yes | Fiscal/report year of the source | Must equal the target year of the run (isolation layer 2) |
| `source_file` | string | yes | Original report/knowledge-base filename | Non-empty |
| `chunk_id` | string | yes | Section/chunk identifier (`section_id`, e.g. `000616::3`) | Non-empty; special value `unattributed` allowed only for fully untraceable conclusions (FR-006) |
| `page_range` | string | yes | Page range, e.g. `p12-15` | Non-empty |
| `doc_type` | string | yes | `report` or `knowledge_base` | Must match the true source type (isolation layer 3); only one type per citation |
| `excerpt` | string | no | Leading text snippet from the source fragment | Max 100 characters (FR-008) |

**Derivation**: Populated exclusively from the metadata of the source `Document Fragment` (an existing section object) the conclusion was attributed to. Never authored or guessed by an LLM.

### 2. AnalysisItem (NEW)

Wraps a single LLM-generated conclusion with its evidence.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | The conclusion text (highlight / weakness) |
| `citations` | list[Citation] | yes | Source fragments supporting this item; empty only when a Citation marked `chunk_id: "unattributed"` is present |

**Validation**: An item MUST have ≥1 citation, OR exactly one `unattributed` citation with an explanatory note (FR-006). Items with no candidate source are never silently dropped.

### 3. KPIItem (EXTENDED)

Existing entity (name, value, unit, pillar). Adds:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `citations` | list[Citation] | no (default `[]`) | The exact section(s) the KPI regex matched; at least 1 expected for rule-extracted KPIs |

### 4. StrategyItem (EXTENDED)

Existing entity (pillar, term, action, rationale). Adds:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `citations` | list[Citation] | no (default `[]`) | Union of the pillar's weakness citations the strategy addresses |

**Validation**: A strategy's citations MUST be a subset of its pillar's weakness citations (never introduces sources not used in the analysis step).

### 5. PillarAnalysis (EXTENDED)

Existing container (pillar, kpis, raw_summary). Changed fields:

| Field | Type | Change |
|-------|------|--------|
| `highlights` | list[AnalysisItem] | was `list[str]` |
| `weaknesses` | list[AnalysisItem] | was `list[str]` |

**Backward note**: `raw_summary` and `overall_summary` remain plain strings; `overall_summary` MAY surface the union of the three pillars' citations in the rendered report but carries no citation structure itself.

### 6. Document Fragment (existing, READ-ONLY for this feature)

The existing section entity (section_id, title, text, pillar, source_file, page_range, company, report_year, doc_type). This feature only **reads** it — no schema change. `chunk_id` in a Citation maps to `section_id` here.

### 7. Containers (existing, mostly unchanged)

- `DiagnosisResult` — company, source_file, E/S/G `PillarAnalysis`, overall_summary. Container only.
- `SolutionResult` — company, strategies (`list[StrategyItem]`), best_practices, gri_gaps. Container only; `best_practices` keep their existing `source` field.

---

## Validation Rules (from requirements)

| Rule | Source | Applies To |
|------|--------|------------|
| Every highlight/weakness/KPI/strategy carries citations | FR-001 | AnalysisItem, KPIItem, StrategyItem |
| Citation contains company, year, chunk_id, page_range (min) | FR-002 | Citation |
| Isolation: no cross-company / cross-year / report↔KB mixing | FR-004 | Citation (all fields), enforced at attribution time |
| Multiple contributing fragments listed together | FR-005 | citation list of an item |
| Untraceable conclusions labeled `unattributed` with note | FR-006 | Citation.chunk_id, AnalysisItem |
| KB sources structurally distinguishable | FR-007 | Citation.doc_type |
| Excerpt ≤ 100 characters | FR-008 | Citation.excerpt |
| Analysis results unchanged by citation feature | FR-009 | All LLM-produced text fields |

---

## Data Flow / State Transitions

```
Search Agent                        Interpretation Agent                  Solution Agent                          Coordinator
─────────────                        ─────────────────────                 ──────────────                          ─────────────
Document Fragments ──(unchanged)──▶  PillarAnalysis                        SessionResult                            Final outputs
  (section_id, company,             ├─ highlights:   [AnalysisItem+cit.]   ├─ strategies:     [StrategyItem+cit.]   ├─ Markdown (inline citations)
   year, page_range,                 ├─ weaknesses:   [AnalysisItem+cit.]   ├─ best_practices: (existing source)     └─ JSON (sources[] arrays)
   doc_type, text)                   └─ kpis:         [KPIItem+cit.]        └─ gri_gaps:       (rule-based, none)
```

1. **Attribution** (interpret): conclusions of a pillar are scored against that pillar's fragment set; top-3 above threshold become citations.
2. **Inheritance** (solution): strategy citations = union of its pillar's weakness citations.
3. **Isolation guard**: attribution refuses any fragment whose `company`/`report_year`/`doc_type` mismatches the run scope.
4. **Rendering** (coordinator): citations serialized per `contracts/output-json.md` and `contracts/markdown-report.md`.