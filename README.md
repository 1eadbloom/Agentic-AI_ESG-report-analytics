# ESG 報告評估與改進多代理協作系統

四個代理人協作，對企業 ESG／永續報告書進行自動化診斷，產出 E/S/G 分項分析與短中長期改善建議，並為每一項結論附上可追溯的原始文件引用。

本專案發想自我在 AgenticAI 課程期末專題設計的四代理架構；後續實際的系統實作、功能擴充（含引用來源追蹤）與全部程式碼皆由本人獨立完成。

## 系統架構

使用者上傳 PDF
│
▼
[Search Agent] PDF解析 → 清理 → 章節切分 → ChromaDB 向量索引
│
▼
[Interpretation Agent] E/S/G 分類 → LLM 摘要 → KPI萃取 → 強弱項識別 → 來源歸因
│
▼
[Solution Agent] LLM 短/中/長期策略 + RAG 最佳實踐檢索 + GRI缺口分析
│
▼
[Coordinator Agent] LangGraph 流程協調 + 品質控管(QC) + 自動重試
│
▼
outputs/ Markdown 報告 + JSON 資料（含引用來源追蹤）


## 開發方法：Spec-Driven Development

引用來源追蹤功能是用 [GitHub spec-kit](https://github.com/github/spec-kit) 以規格驅動開發（Spec-Driven Development）流程建構的：

/speckit.specify → /speckit.plan → /speckit.tasks → /speckit.implement → /speckit.converge


完整設計文件保留在 [`specs/001-citation-source-tracking/`](specs/001-citation-source-tracking/)（spec.md、plan.md、research.md、data-model.md、contracts/、quickstart.md、tasks.md），可以看到從需求規格到實作任務的完整脈絡。

`/speckit.converge` 在實作完成後重新比對程式碼與規格，實際抓到一個測試沒覆蓋到的真實邊界案例：ChromaDB 檢索路徑（增量更新、檔案未變更時觸發）遺失了 `chunk_id`，導致該路徑產生的引用缺少溯源資訊。這個落差被自動記錄為新任務並修復、驗證，過程完整保留在 `tasks.md` 的 Phase 7。

## 引用來源追蹤（Citation Source Tracking）

最終輸出的每一項結論（亮點 / 待改善 / KPI / 改善策略）都會附上**來源引用**，讓使用者可以回頭驗證 AI 輸出的依據：

- **Markdown**：每一項後方內嵌 `(來源: {公司} · {年度} · chunk {chunk_id} · {頁碼}：「摘錄」)`；
  無法溯源的一般性結論標記 `(非文件依據 · 通用知識或經驗判斷…)`。
- **JSON**：每個項目新增 `sources[]` 陣列，含 `company`、`report_year`、
  `source_file`、`chunk_id`、`page_range`、`doc_type`、`excerpt`（≤100 字）。
- **隔離保證**：引用只會指向本次執行公司、同年度、`report` 類型的片段；
  KB 來源與公司報告來源不會混用。
- 歸因為**確定性方法**（embedding 相似度 top-k + KPI 規則精確溯源），不要求 LLM
  自我回報來源，也不改變原本的分析結果。
-已用真實上市公司永續報告書（PDF）驗證過完整端到端流程，實際產出範例見
  [`examples/中信證券_ESG分析報告.md`](examples/中信證券_ESG分析報告.md)。

## 測試

```bash
pip install -r requirements.txt
pytest -q
```

測試套件（40 項，unit + integration）**完全離線執行**，不需要下載 embedding 模型、不需要 LLM API、不需要 ChromaDB——全部用假的（fake）embedding 與 mock LLM 驗證邏輯正確性，幾秒內跑完：

- `tests/unit/`：Citation 資料結構、確定性歸因邏輯、Markdown/JSON 渲染契約、隔離守門
- `tests/integration/`：跨公司/跨年度隔離保證、端對端管線（mocked LLM）契約驗證

## 快速開始（完整跑一次真實分析）

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 設定環境
cp .env.example .env

# 3. 把 ESG 報告 PDF 放進 data/reports/
#    （本 repo 不含真實企業報告 PDF，需自行取得公開的永續報告書）

# 4. 執行分析（第一次會下載 embedding 模型，約 420MB）
python coordinator_agent.py --file "2024年永續報告書_中_中信證券000616.pdf" --company "中信證券"

# 5. 查看報告
ls outputs/
```

## 新增報告

直接把新的 PDF 放進 `data/reports/`，再次執行即可（自動增量更新）：
```bash
python coordinator_agent.py --file "新公司_2024_ESG.pdf" --company "新公司"
```

## 主要參數

| 參數 | 說明 | 預設 |
|------|------|------|
| `--file` | PDF 檔名（data/reports/ 中） | 必填 |
| `--company` | 公司名稱 | 從檔名推斷 |
| `--model` | LiteLLM 模型名稱（需為本機 Ollama 已 pull 的模型，用 `ollama list` 確認） 
| `ollama/gemma3:4b` || `--rebuild` | 重建 ChromaDB 索引 | False |

## 知識庫管理

`data/knowledge_base/` 中放置參考文件（GRI 標準、最佳實踐、產業資料），
系統會自動將這些文件也納入 ChromaDB，讓 Solution Agent 在生成建議時可以檢索引用。

新增知識庫文件後，重新執行 Search Agent：
```bash
python search_agent.py --rebuild
```

## 輸出說明

- `outputs/公司名稱_ESG分析報告_日期.md`：人類可讀的 Markdown 報告
- `outputs/公司名稱_ESG分析報告_日期.json`：結構化 JSON（供程式化後處理）

## 已知限制

- **章節選取的相關性排序**：各 E/S/G 面向候選片段超過 40 個時，會先用 embedding 相似度排序後取前 40 個送入 LLM 分析，而非直接使用全部片段——對篇幅極大（數百頁）的報告，理論上仍可能有相關內容排在門檻之外而未被納入分析。
- **KPI 規則式擷取（正則表達式）** 在 PDF 轉文字後版面錯亂或中英夾雜的段落上，偶爾會擷取到不具意義的片段（例如誤將版面元素當成指標名稱）。這不影響 citation 機制本身的正確性，但會讓最終 KPI 清單中混入少量雜訊，尚待更嚴謹的規則或後處理過濾。
- **策略引用的去重上限**：改善策略透過「繼承其面向所有弱點的引用」產生 sources，目前沒有數量上限，弱點來源多時單條策略的引用列表可能偏長。
- **小型 LLM（4B 等級）的格式穩定性**：分析文字由本機小型模型生成時，偶爾不會嚴格遵循 prompt 要求的段落格式，導致個別面向的亮點/待改善退回「未包含相關資訊」的保底文字；此為 LLM 輸出穩定性問題，非程式邏輯錯誤，使用參數量更大的模型可改善。
- Constitution（`.specify/memory/constitution.md`）尚未填入專案特定的治理原則，目前為未初始化的範本狀態。