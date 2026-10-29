"""
test_ollama_direct.py — 繞過 LiteLLM，直接呼叫 Ollama，確認 think=false 是否真的讓 token 變少

用法：python test_ollama_direct.py
比較兩個數字：think=True 與 think=False 的 eval_count（輸出 token 數）。
"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")   # 避免 Windows cp950 編碼問題

MODEL = "gemma4-plan:latest"
URL = "http://localhost:11434/api/chat"

PROMPT = """你是「中信證券」的企業永續策略顧問。
以下是「治理（G）」面向的 ESG 診斷待改善事項：
  - 供應鏈後續管理機制尚待明晰，需補充輔導流程

請只針對「治理（G）」面向提出 5 項具體可行的改善策略：
- 短期（1年內）：2 項
- 中期（2-3年）：2 項
- 長期（3-5年）：1 項

輸出規則：只輸出這 5 行，不要標題、說明、空行或 Markdown 符號；
每行格式為「[G][期程] 行動：理由」，期程只能是「短期」「中期」「長期」。
"""


def run(think: bool) -> None:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一位專業的 ESG 永續顧問。"},
            {"role": "user", "content": PROMPT},
        ],
        "stream": False,
        "think": think,
        "options": {"num_predict": 4096},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read().decode("utf-8"))
    msg = data.get("message", {})
    print(f"--- think={think} ---")
    print("耗時（秒）:", round(time.time() - t0, 1))
    print("輸出 token（eval_count）:", data.get("eval_count"))
    print("done_reason:", data.get("done_reason"))
    print("content 長度:", len(msg.get("content") or ""))
    print("thinking 長度:", len(msg.get("thinking") or ""))
    print("content 開頭:", repr((msg.get("content") or "")[:120]))
    print()


if __name__ == "__main__":
    run(think=True)
    run(think=False)
