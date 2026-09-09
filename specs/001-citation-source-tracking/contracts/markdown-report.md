# Contract: Markdown Report Citation Rendering

**Date**: 2026-09-08 | **Feature**: [spec.md](../spec.md) | **Readability target**: [SC-005](../spec.md)

This contract defines how citations appear in the human-readable Markdown report (`outputs/*_ESG分析報告_*.md`). It is the user-facing verification interface.

## Citation string format

Both report and knowledge-base sources render as a compact inline parenthetical after the conclusion. The **first** source includes a quoted excerpt (US3); remaining sources show metadata only, to keep the line readable (SC-005):

```
(來源: {company} · {report_year} · chunk {chunk_id} · {page_range}：「{excerpt}」)
```

- Multiple sources join with ` ; ` (semicolon+space):
  ```
  (來源: 中信證券 · 2024 · chunk 000616::3 · p12-15：「本公司已完成 2024 年度溫室氣體盤查…」 ; 來源: 中信證券 · 2024 · chunk 000616::5)
  ```
- **Knowledge-base** sources are visually distinct — prefixed with `KB`:
  ```
  (KB 來源: 知識庫 · chunk gri_standards_overview.md::2 · p1：「GRI 標準要求組織揭露邊界…」)
  ```
- **Unattributed** conclusion (FR-006):
  ```
  (非文件依據 · 通用知識或經驗判斷，非特定文件片段)
  ```

**Excerpt rule**: excerpt is stored up to 100 chars (FR-008, full value in JSON). In Markdown it displays truncated to 60 chars with an ellipsis (`…`); it is only appended to the first listed source of each conclusion.

## Where it renders

| Output section | Rendered on |
|----------------|-------------|
| 整體摘要 (overall summary) | Below the summary — an optional consolidated citation line listing the union of pillar citations of the top ≥1 supporting fragment |
| E/S/G sections — 亮點 Highlights | Each bullet, inline after the text |
| E/S/G sections — 待改善 Weaknesses | Each bullet, inline after the text |
| E/S/G sections — 關鍵績效指標 KPIs (`kpis[:5]`) | Each KPI line, inline after the value+unit |
| 📋 改善策略建議 strategies | Each strategy bullet, inline after the action/rationale |
| 🏆 同產業最佳實踐 | Keeps existing `> 來源: {source}` blockquote (unchanged) |
| 🔍 GRI/TCFD 對標缺口分析 | None (rule-based) |

## Rules

1. **Inline, not footnotes** — a citation must sit on the same visual line as the conclusion it supports (verifiability in place, per research R4).
2. Citations must never break the bullet text mid-statement; place at end of the item.
3. No concordance/reference section is introduced; the report remains self-contained.
4. A conclusion with `chunk_id: "unattributed"` renders the unattributed note, never an empty `()`.
5. Rendering adds no additional LLM or retrieval calls (SC-004).

## Example (rendered)

```markdown
- 已達 2024 年溫室氣體盤查 ISO 14064 確信，年度排放較前年下降 12%。
  (來源: 中信證券 · 2024 · chunk 000616::7 · p12-15：「本公司已完成 2024 年度溫室氣體盤查…」)
- 董事會成員對永續治理之監督職責明確，獨立董事比例達 1/3。
  (來源: 中信證券 · 2024 · chunk 000616::11 · p31-33：「審計委員會成員皆具財務與永續專長…」)
- 短期建議導入 ISO 50001 能源管理系統，並將綠電比例納入年度 KPI。
  (來源: 中信證券 · 2024 · chunk 000616::7 · p12-15：「能源管理現況說明…」 ; 來源: 中信證券 · 2024 · chunk 000616::9)
```