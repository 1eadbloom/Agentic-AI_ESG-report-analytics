"""
test_litellm_think.py — 找出「透過 LiteLLM」關閉 Ollama 思考的有效寫法

直接呼叫 Ollama 時 think=False 只需約 226 token；這裡逐一測試 LiteLLM 的不同寫法，
看哪一種能讓 completion_tokens 降到幾百。數字越小代表思考真的被關掉。

用法：python test_litellm_think.py
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")   # 避免 Windows cp950 編碼問題

from litellm import completion

BASE = "http://localhost:11434"
MODEL = "gemma4-plan:latest"
MESSAGES = [
    {"role": "system", "content": "你是一位專業的 ESG 永續顧問。"},
    {"role": "user", "content": (
        "針對「治理（G）」面向的待改善事項「供應鏈後續管理機制尚待明晰」，"
        "請提出 5 項改善策略，每行格式為「[G][期程] 策略名稱：理由」，"
        "期程只能是短期、中期、長期，只輸出這 5 行。")},
]

VARIANTS = {
    "A. ollama/  + extra_body think=False（你目前的寫法）":
        dict(model=f"ollama/{MODEL}", extra_body={"think": False}),
    "B. ollama_chat/ + extra_body think=False":
        dict(model=f"ollama_chat/{MODEL}", extra_body={"think": False}),
    "C. ollama_chat/ + think=False 參數":
        dict(model=f"ollama_chat/{MODEL}", think=False),
    "D. ollama_chat/ + reasoning_effort='disable'":
        dict(model=f"ollama_chat/{MODEL}", reasoning_effort="disable"),
    "E. ollama_chat/ + 不加任何參數（對照組）":
        dict(model=f"ollama_chat/{MODEL}"),
}

for name, kw in VARIANTS.items():
    t0 = time.time()
    try:
        r = completion(messages=MESSAGES, api_base=BASE, max_tokens=4096, **kw)
        used = r["usage"]["completion_tokens"]
        content = (r["choices"][0]["message"]["content"] or "")
        print(f"{name}\n   completion_tokens={used}  耗時={time.time()-t0:.1f}s  "
              f"content 長度={len(content)}\n")
    except Exception as e:   # 某些寫法可能不被支援，直接顯示原因
        print(f"{name}\n   失敗：{type(e).__name__}: {str(e)[:150]}\n")
