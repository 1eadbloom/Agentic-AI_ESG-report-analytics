"""
ESG multi-agent 專案 — Citation Faithfulness / Attribution Accuracy 評分腳本
================================================================

用途:
    針對 citation source tracking 功能，用 15-20 組已知答案的 Q&A，
    量化模型輸出跟 golden answer 之間的語意相似度，
    算出一個可以在面試時報出來的 faithfulness / attribution accuracy 分數。

使用方式:
    1. 準備 golden_qa.json（見下方範例格式），從中信證券 2024 永續報告
       手動整理 15-20 組「問題 + 正確答案(可摘自報告原文) + 對應章節/pillar」。
    2. 把你的 ESG multi-agent pipeline 針對同樣的 15-20 個問題各跑一次，
       記錄模型輸出與它標註的 citation 章節。
    3. 執行: python esg_faithfulness_eval.py golden_qa.json model_outputs.json

golden_qa.json 範例格式:
[
  {
    "id": "q1",
    "question": "公司2024年的溫室氣體排放範疇一總量是多少?",
    "golden_answer": "...(從報告原文摘錄的正確答案)...",
    "golden_section": "環境-溫室氣體排放"
  },
  ...
]

model_outputs.json 範例格式（跑你自己的 pipeline 後產出）:
[
  {
    "id": "q1",
    "model_answer": "...(模型實際輸出)...",
    "cited_section": "...(模型標註的來源章節，來自你的 citation.py)..."
  },
  ...
]

輸出:
    - 每題的 answer similarity 分數與 section 是否命中
    - 整體的平均 faithfulness 分數（0-1）
    - 整體的 attribution accuracy（citation 章節命中率）

注意:
    這個腳本假設你已經在 ESG 專案的 venv 裡裝了 sentence-transformers。
    模型用你原本 pipeline 已經在用的 sentence-transformers 模型即可，
    這樣分數才跟你 pipeline 本身的檢索品質一致，不是換一套標準。
"""

import json
import sys
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("請先在你的 ESG 專案 venv 裡安裝: pip install sentence-transformers")
    sys.exit(1)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(golden_path: str, model_output_path: str, model_name: str = "all-MiniLM-L6-v2"):
    golden = {item["id"]: item for item in load_json(golden_path)}
    outputs = {item["id"]: item for item in load_json(model_output_path)}

    missing = set(golden) - set(outputs)
    if missing:
        print(f"警告: 以下題目在 model_outputs.json 裡找不到對應輸出: {missing}")

    model = SentenceTransformer(model_name)

    results = []
    for qid, g in golden.items():
        if qid not in outputs:
            continue
        o = outputs[qid]

        emb_golden = model.encode(g["golden_answer"], convert_to_tensor=True)
        emb_model = model.encode(o["model_answer"], convert_to_tensor=True)
        similarity = util.cos_sim(emb_golden, emb_model).item()

        section_hit = g.get("golden_section") == o.get("cited_section")

        results.append({
            "id": qid,
            "question": g["question"],
            "similarity": round(similarity, 4),
            "section_hit": section_hit,
        })

    if not results:
        print("沒有可評估的題目，請檢查兩個檔案的 id 是否對得上。")
        return

    avg_faithfulness = sum(r["similarity"] for r in results) / len(results)
    attribution_accuracy = sum(1 for r in results if r["section_hit"]) / len(results)

    print("\n=== 逐題結果 ===")
    for r in results:
        hit = "✓" if r["section_hit"] else "✗"
        print(f"[{r['id']}] similarity={r['similarity']:.3f}  section_hit={hit}  {r['question']}")

    print("\n=== 整體分數（面試可直接報這兩個數字）===")
    print(f"平均 Faithfulness (embedding similarity): {avg_faithfulness:.3f}")
    print(f"Attribution Accuracy (citation章節命中率): {attribution_accuracy:.1%}")

    out_path = Path(model_output_path).with_name("faithfulness_eval_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_question": results,
            "avg_faithfulness": avg_faithfulness,
            "attribution_accuracy": attribution_accuracy,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存到: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python esg_faithfulness_eval.py golden_qa.json model_outputs.json")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
