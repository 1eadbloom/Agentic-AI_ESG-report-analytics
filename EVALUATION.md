# Evaluation

本文件記錄本專案的評測方法與結果，重點是 citation source tracking 功能的可信度驗證。

## 為什麼不是傳統的 golden Q\&A 評測

一開始本專案的`coordinator\_agent.py` 其設計的運作方式即是「輸入一份
PDF 報告，一次性輸出整份 E/S/G 分析（亮點／待改善／KPI／改善策略，各自附帶 citation）」，方便一次性檢核不同面向與整體的檢索品質，而非「單次輸入任意問題並即時作答」的 QA 系統。

因此本次品質檢核設計即為**直接評測 pipeline 自己產出的每一項結論，跟它所標註的 citation 原文片段之間，語意是否真的對得上**——這個設計更貼合系統實際的功能（citation 是否可信），也不需要額外維護多組 golden Q\&A的產出。

## 方法論

對 pipeline 針對中信證券 2024 年永續報告書的一次完整分析輸出：

1. 攤平出所有「結論文字（claim）— citation 摘錄（excerpt）」配對
2. 用多語言 sentence-transformers 模型（`paraphrase-multilingual-MiniLM-L12-v2`）
計算每組 claim 與其 excerpt 的 embedding cosine similarity
3. 建立對照組（baseline）：把 claim 與「別組的」excerpt 隨機配對，模擬 citation
選錯段落時的分數分布，作為「真正文不對題」的參考基準
4. 對分數偏低（<0.5）的組別，額外用本地 LLM 做「是／否支持」的二元判斷，
產出一個更容易對非技術受眾溝通的支持率指標

單一連續分數本身不足以說明 citation 機制是否有效——必須跟對照組比較，兩者的差距
（uplift）才是機制有效性的證據。這是評測設計上刻意的取捨。

## 評測過程中的迭代

第一版結果 uplift 只有 +0.038，幾乎看不出真實 citation 比隨機配對好多少。追查後
發現三個問題，依序修正：

|問題|根因|修正|
|-|-|-|
|少數結論是 `###`、`．` 等 markdown 符號或孤立標點|上游解析 LLM 原始輸出時，偶爾將格式符號誤判為正式結論項目|評測腳本中過濾內容長度 < 5 字元的項目|
|「分析結果未包含改善建議」這類系統防呆佔位文字被當成正常結論，一起送進 citation 比對|`parse\_section()` 在 LLM 未生成對應區塊時，用這段文字當 fallback，但沒有在產生 citation 前排除它|評測腳本中過濾已知的佔位文字樣式|
|换多語言 embedding 模型前後，uplift 差距達 3 倍|原本用的 `all-MiniLM-L6-v2` 是英文語料為主的模型，對中文語意鑑別力弱，真假配對分數被壓縮在相近區間|改用 `paraphrase-multilingual-MiniLM-L12-v2`|

## 最終結果（45 組結論－citation 配對）

* 真實 citation 平均相似度：**0.589**
* 隨機配對對照組平均相似度：**0.460**
* Uplift（真實 citation 相似度顯著高於對照組）：**+0.128**
* 相似度低於 0.3（明顯文不對題）的 citation 數：**0 / 45（0%）**
* LLM 二元判斷支持率：**37 / 45（82.2%）**

### 分類 breakdown

|項目類型|平均分數|樣本數|
|-|-|-|
|highlights|0.607|27|
|weaknesses|0.561|18|

|E/S/G 面向|平均分數|樣本數|
|-|-|-|
|E|0.626|21|
|G|0.569|21|
|S|0.464|3|

`weaknesses` 類結論分數略低於 `highlights`，可能的原因是「待改善」類結論常是在指出
報告中缺乏的資訊，本質上比複述具體事實的 highlight 更難找到一段精確對應的原文佐證，
屬於這類分析的特性，不一定是 citation 機制的缺陷。S 面向樣本數偏少（3 組），單次結果
的代表性有限，後續可考慮增加測試文件來擴大樣本。

## 已知限制

* 評測樣本來自單一份報告（中信證券 2024 永續報告書）的單次執行，尚未對多份不同公司/
年度的報告重複驗證穩定性
* 評測過程中同步發現並修正了兩個與評測無直接關聯的 pipeline bug（KPI 欄位 `unit`
在特定解析路徑下為 `None`、防呆佔位文字誤入正常資料流程），這兩個 bug 不影響
本次評測數字的有效性（評測腳本已過濾相關髒資料），但屬於產品本身待修復項目

## 執行方式

```bash
# 1. 產出分析報告（需本地 LLM 服務已啟動）
python coordinator\_agent.py --file "<報告PDF路徑>" --company "<公司名稱>"

# 2. 對輸出的 JSON 跑評測
python eval/citation\_faithfulness\_eval.py outputs/<公司>\_ESG分析報告\_<日期>.json
```

