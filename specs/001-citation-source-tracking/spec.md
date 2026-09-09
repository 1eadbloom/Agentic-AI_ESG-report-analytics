# Feature Specification: Citation Source Tracking

**Feature Branch**: `001-citation-source-tracking`

**Created**: 2026-09-08

**Status**: Draft

**Input**: User description: "在ESG報告分析系統的最終輸出加入引用來源追蹤功能。當Coordinator agent產生最終結論時，每一段結論都要附上它是根據哪些原始文件片段得出的，包含來源公司名稱、年度、以及對應的文件chunk id或段落位置，讓使用者可以回頭驗證AI輸出的依據，同時不能破壞既有的三層metadata隔離機制（跨公司、跨年度資料不能互相污染）。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Verify a Specific Conclusion's Source (Priority: P1)

As an ESG analyst reviewing the AI-generated report, I want to see exactly which original document fragments each conclusion is based on, so that I can verify the AI's reasoning and trust the output.

**Why this priority**: This is the core value proposition. Without traceable citations, users cannot validate the AI's analysis, undermining the system's credibility for decision-making.

**Independent Test**: Can be fully tested by generating a report for a single company and verifying that every highlight, weakness, KPI, and strategy in the output includes source citations with company name, year, and chunk ID.

**Acceptance Scenarios**:

1. **Given** the system has analyzed a company's ESG report, **When** the user views the Markdown report, **Then** every highlight item under E/S/G sections displays the source company name, report year, and at least one chunk ID or paragraph position.
2. **Given** the system has analyzed a company's ESG report, **When** the user views the JSON output, **Then** each diagnosis finding and each strategy includes a `sources` array containing objects with `company`, `year`, `chunk_id`, and `page_range` fields.
3. **Given** a conclusion is derived from multiple document fragments, **When** the user views that conclusion, **Then** all contributing source fragments are listed (not just one).

---

### User Story 2 - Cross-Reference Citations Across Companies (Priority: P2)

As a portfolio analyst comparing multiple companies, I want to see source citations that clearly distinguish which company and year each fragment comes from, so that I can confirm no data from Company A was incorrectly used to draw conclusions about Company B.

**Why this priority**: The three-layer metadata isolation is a critical system constraint. This story validates that citations reinforce rather than compromise data isolation.

**Independent Test**: Can be tested by analyzing reports from two different companies and verifying that no citation in Company A's report references Company B's document fragments.

**Acceptance Scenarios**:

1. **Given** the system has analyzed reports from Company A (2024) and Company B (2023), **When** the user views Company A's report, **Then** every citation references only Company A's document fragments with year 2024.
2. **Given** the system has analyzed reports from multiple companies, **When** the user inspects the JSON output, **Then** the `company` field in each citation matches the report's target company.

---

### User Story 3 - Trace a Citation Back to Original Text (Priority: P3)

As a compliance officer, I want to see a brief excerpt of the original source text alongside each citation, so that I can quickly verify the AI's interpretation without opening the original PDF.

**Why this priority**: This enhances usability but is not strictly required for the citation tracking to function. Users can still look up chunks by ID manually.

**Independent Test**: Can be tested by generating a report and confirming that citations include a short text snippet from the original document fragment.

**Acceptance Scenarios**:

1. **Given** a conclusion cites a specific chunk, **When** the user views the citation in the Markdown report, **Then** a brief excerpt (up to 100 characters) of the original source text is displayed alongside the citation metadata.
2. **Given** a conclusion cites a specific chunk, **When** the user views the JSON output, **Then** the citation object includes an `excerpt` field containing the source text snippet.

---

### Edge Cases

- What happens when the LLM generates a conclusion that cannot be traced to any specific document fragment (e.g., a generic industry statement)?
  - The system SHOULD mark such conclusions with a `sources` array containing a single entry with `chunk_id: "unattributed"` and a note indicating the conclusion is based on general knowledge rather than specific document evidence.
- What happens when a document chunk is used to derive conclusions for multiple E/S/G pillars?
  - The same chunk ID SHOULD appear in citations for each pillar it contributed to. Deduplication is not required at the citation level.
- What happens when the source document chunk has been removed or the file hash has changed since analysis?
  - Citations reference the chunk ID as it existed at analysis time. The system does not need to validate that the chunk still exists in the vector store at display time.
- What happens when knowledge base documents (not company reports) contribute to a conclusion?
  - Citations SHOULD distinguish between report sources (`doc_type: "report"`) and knowledge base sources (`doc_type: "knowledge_base"`).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST attach source citations to every highlight, weakness, KPI, and strategy item in the final report output.
- **FR-002**: Each citation MUST include at minimum: source company name, report year, chunk ID (section identifier), and page range.
- **FR-003**: Citations MUST be included in both Markdown and JSON output formats.
- **FR-004**: System MUST preserve the three-layer metadata isolation: citations for Company A MUST NOT reference document fragments from Company B; citations for year 2024 MUST NOT reference year 2023 fragments; report citations MUST NOT reference knowledge base fragments unless explicitly annotated as knowledge base sources.
- **FR-005**: When a conclusion is derived from multiple source fragments, all contributing fragments MUST be listed in the citation.
- **FR-006**: Conclusions that cannot be traced to specific document fragments MUST be labeled as unattributed with an explanatory note.
- **FR-007**: Citations for knowledge base sources MUST be visually or structurally distinguishable from report sources.
- **FR-008**: System MUST include a brief text excerpt (up to 100 characters) from the original source fragment in each citation.
- **FR-009**: The citation mechanism MUST NOT alter the existing analysis pipeline's behavior — conclusions generated without citations in a prior step must still produce the same analytical output.

### Key Entities

- **Citation**: A reference linking an AI-generated conclusion to its source document fragment. Key attributes: company name, report year, chunk ID, page range, document type, source text excerpt.
- **Document Fragment** (existing): Source document segment with section_id, title, text, pillar, source_file, page_range, company, report_year, doc_type. Citations are derived from the metadata of these segments.
- **DiagnosisResult** (existing): Analysis output containing highlights, weaknesses, KPIs per pillar. Each finding item will be extended to carry citation metadata.
- **SolutionResult** (existing): Strategy recommendations. Each strategy will be extended to carry citation metadata.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of non-unattributed conclusions in the final report include at least one valid source citation.
- **SC-002**: Zero cross-company citation contamination — no citation in Company A's report references Company B's document fragments.
- **SC-003**: Users can identify the specific source document, page, and chunk for any conclusion in the report within 5 seconds of viewing it.
- **SC-004**: The citation tracking mechanism does not increase report generation time by more than 10% compared to the current baseline.
- **SC-005**: The Markdown report remains human-readable and well-formatted with citations integrated naturally alongside conclusions.

## Assumptions

- The existing document fragment entity already carries company, report_year, source_file, section_id, and page_range metadata, which provides the foundation for building citations.
- The Interpretation Agent and Solution Agent currently lose the mapping between their outputs and the specific input sections that informed them. This feature assumes that re-establishing this mapping is feasible by propagating section metadata through the analysis pipeline.
- The three-layer metadata isolation (cross-company, cross-year, report vs. knowledge base) is already enforced at the data retrieval layer and does not need to be reimplemented, only respected by the citation mechanism.
- Citations are generated deterministically based on which sections were passed to each analysis step, not by asking the LLM to self-report its sources (which would be unreliable).
- Users of this system are ESG analysts or compliance officers with sufficient domain knowledge to interpret chunk IDs and page references.
- The Markdown output format will use inline parenthetical citations (e.g., `(來源: 中信證券, 2024, chunk: 000616::3, p12-15)`) to maintain readability.
- The JSON output will include structured citation objects for programmatic access.
