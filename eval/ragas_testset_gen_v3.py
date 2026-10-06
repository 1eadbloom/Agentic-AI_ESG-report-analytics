"""
ESG multi-agent 專案 — 用 RAGAS TestsetGenerator 從真實報告PDF自動產生 golden Q&A
================================================================

這版改用 Ollama 本地伺服器（gemma4-plan:latest），取代之前的LM Studio。
換的原因：35B模型在LM Studio上反覆出現崩潰/連線中斷，懷疑是硬體資源
（尤其系統RAM）撐不住長時間CPU運算導致的不穩定。換一顆較小的模型，
應該能大幅改善穩定性跟速度。

安裝:
    pip install ragas langchain-community langchain-ollama langchain-text-splitters pypdf unstructured langchain-huggingface

用法:
    python ragas_testset_gen_v3.py path/to/your_esg_report.pdf 20
    （執行前先確認 Ollama 服務已啟動：ollama serve，
     且 gemma4-plan:latest 已經 pull 下來：ollama list 確認看得到）

備註:
    - 這裡接的是 Ollama 的原生API（http://localhost:11434），不是LM Studio的
      OpenAI相容端點，兩者client class不能共用，這版換回 ChatOllama。
    - embedding繼續用本地 sentence-transformers（跟LLM選哪顆模型無關，
      不用因為換了LLM就跟著換embedding）。
    - 如果這顆模型的品質/生成格式不穩定（小模型有時候JSON格式會出錯），
      屬於已知取捨——先求跑得完、跑得穩，生成完一定要人工過濾，
      不要直接全部拿去當golden set用。
    - PDF若是掃描檔或版面複雜（表格多），建議用 UnstructuredPDFLoader
      而不是 PyPDFLoader，抽取品質會好很多，但速度較慢。
"""

import sys
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Ollama 本地伺服器，預設位址跟port。
OLLAMA_BASE_URL = "http://localhost:11434"
# 模型名稱要跟 `ollama list` 裡顯示的tag完全一致（含冒號後的版本標籤）。
# 不需要另外建nothink版本的模型，思考模式是在下面用reasoning=False
# 在Python端關閉的。
OLLAMA_CHAT_MODEL = "gemma4-plan:latest"


def main(pdf_path: str, testset_size: int = 20):
    # Step 1: 讀真實報告 PDF
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    print(f"讀到 {len(raw_docs)} 頁，來自: {pdf_path}")

    # chunk_size先保守抓1500字元。換了模型後，如果這顆模型的實際context
    # window比之前大/小，可以再依照 `ollama show gemma4-plan:latest` 顯示的
    # context length等比例調整，抓 token數 × 0.3 當chunk_size字元數上限。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "，", " ", ""],  # 中文常用句讀優先切
    )
    docs = splitter.split_documents(raw_docs)
    print(f"切成 {len(docs)} 個chunk後再送進ragas")

    # 強烈建議：先用少量chunk小跑一次確認pipeline走得通，再跑全量。
    # 測試時可以把下面這行取消註解，只拿前幾個chunk試跑：
    # docs = docs[:50]

    # Step 2: 設定 generator 用的 LLM（接 Ollama）跟 embedding（本地sentence-transformers）
    # reasoning=False是langchain-ollama官方文件確認的正確參數名稱：
    # 關掉模型的思考模式，不會再把<think>...</think>推理過程塞進主要回應內容，
    # 避免把num_predict輸出上限吃光、也避免ragas的JSON parser去解析到推理文字。
    # 注意：think不是Modelfile的PARAMETER（那是API呼叫層級的設定），
    # 所以這裡是在Python端設定，不需要額外建立nothink版本的模型。
    # format="json"：強制Ollama用語法限制(constrained decoding)逼模型只能輸出
    # 合法JSON語法，從根源避免模型把摘要寫成一般英文段落、沒包進JSON物件的狀況
    # （這就是你上次327個chunk跑到最後，少數幾個node卡在這裡導致整批失敗的原因）。
    ollama_llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, reasoning=False, format="json"
    )
    local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    generator_llm = LangchainLLMWrapper(ollama_llm)
    generator_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

    # Step 3: 生成（generate_with_chunks：直接用我們自己切好的chunk，
    # 跳過ragas內部的標題抽取/切分，自動處理summary/embedding/persona等步驟）
    # max_workers=1：避免本地伺服器同時處理多個請求時，context被拆分
    # 給多個並行請求、導致單一請求實際可用的context變小的問題。
    run_config = RunConfig(max_workers=1)

    dataset = generator.generate_with_chunks(
        chunks=docs, testset_size=testset_size, run_config=run_config
    )

    df = dataset.to_pandas()
    print(f"\n共生成 {len(df)} 題，欄位: {list(df.columns)}")

    # Step 4: 轉成你 esg_faithfulness_eval.py 吃的 golden_qa.json 格式
    golden_qa = []
    for i, row in df.iterrows():
        golden_qa.append({
            "id": f"q{i+1}",
            "question": row.get("user_input", ""),
            "golden_answer": row.get("reference", ""),
            "golden_section": "【填入：對照報告目錄手動標註章節】",
            "_reference_contexts": row.get("reference_contexts", []),  # 供你人工核對用
        })

    out_path = Path(pdf_path).with_name("ragas_generated_golden_qa.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden_qa, f, ensure_ascii=False, indent=2)
    print(f"\n已存成: {out_path}")
    print("下一步：人工過濾/修正這份清單（品質不佳的題目直接刪），")
    print("並補上 golden_section，才能拿去跑 esg_faithfulness_eval.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python ragas_testset_gen_v3.py path/to/report.pdf [testset_size]")
        sys.exit(1)
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    main(sys.argv[1], size)
