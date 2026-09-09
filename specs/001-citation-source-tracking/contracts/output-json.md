# Contract: Structured JSON Output (`outputs/*_ESG分析報告_*.json`)

**Date**: 2026-09-08 | **Feature**: [spec.md](../spec.md) | **Data model**: [data-model.md](../data-model.md)

This contract defines the machine-readable JSON output emitted by `coordinator_agent.py`. It is the programmatic post-processing interface; changes are additive-only (existing fields below marked "existing" are unchanged).

## Top-level object

```json
{
  "company": "<string> /* existing */",
  "diagnosis": { "<E|S|G>": { "...": "see PillarAnalysis" }, "overall_summary": "<string> /* existing */" },
  "solution": { "strategies": [], "best_practices": [], "gri_gaps": [] },
  "errors": ["<string>", "..."] /* existing */
}
```

## Citation object

A `source` / `sources[]` element:

```json
{
  "company": "<source company name>",
  "report_year": "<fiscal year string>",
  "source_file": "<original file name>",
  "chunk_id": "<section id> | \"unattributed\"",
  "page_range": "p12-15",
  "doc_type": "report | knowledge_base",
  "excerpt": "<string, max 100 chars, optional>"
}
```

**Special value**: `chunk_id == "unattributed"` marks an untraceable conclusion; the companion `company`/`report_year` fields hold the run scope and `source_file` is empty (`""`).

## PillarAnalysis (per pillar `E` / `S` / `G`)

```json
{
  "pillar": "E",
  "highlights": ["<AnalysisItem>", "..."],   /* CHANGED: objects instead of strings */
  "weaknesses": ["<AnalysisItem>", "..."],    /* CHANGED: objects instead of strings */
  "kpis": ["<KPIItem>", "..."],               /* EXTENDED: each has sources[] */
  "raw_summary": "<string> /* existing */"
}
```

### AnalysisItem

```json
{
  "text": "<conclusion text>",
  "sources": ["<Citation object>", "..."]     /* ≥1 expected, unless a single unattributed entry */
}
```

### KPIItem

```json
{
  "name": "<string> /* existing */",
  "value": "<string> /* existing */",
  "unit": "<string | null> /* existing */",
  "pillar": "<string | null> /* existing */",
  "sources": ["<Citation object>", "..."]     /* NEW, default [] */
}
```

## Solution objects

### StrategyItem

```json
{
  "pillar": "<string> /* existing */",
  "term": "short | mid | long /* existing */",
  "action": "<string> /* existing */",
  "rationale": "<string> /* existing */",
  "sources": ["<Citation object>", "..."]     /* NEW, default [] */
}
```

**Invariant**: a strategy's `sources` ⊆ union of its pillar's weakness `sources`.

### best_practices

```json
{
  "pillar": "<string> /* existing */",
  "suggestion": "<string> /* existing */",
  "source": "<knowledge-base reference> /* existing, unchanged */"
}
```

### gri_gaps

`["<string>", "..."]` — rule-based, **no** `sources` (unchanged).

## Contract assertions (used by tests)

1. Every element of `highlights`, `weaknesses`, `kpis`, and `strategies` contains a `sources` key.
2. Every non-`unattributed` Citation has non-empty `company`, `report_year`, `source_file`, `chunk_id`, `page_range`; `excerpt` ≤ 100 chars.
3. Isolation (SC-002): for run with target company `C` and year `Y`, every Citation's `company == C` and `report_year == Y`; no `doc_type` mixing within a conclusion's source list unless an explicit `knowledge_base` entry is structurally distinct.
4. Additive stability: fields marked "existing"/"unchanged" retain their prior types and presence.