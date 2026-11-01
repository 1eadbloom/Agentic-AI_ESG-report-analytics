import sys
sys.stdout.reconfigure(encoding="utf-8")   # 避免 Windows cp950 編碼問題
from litellm import completion

prompt = """你是「中信證券」的企業永續策略顧問。
以下是「治理（G）」面向的 ESG 診斷待改善事項：
  - 供應鏈後續管理機制尚待明晰，需補充輔導流程

請只針對「治理（G）」面向提出 5 項具體可行的改善策略：
- 短期（1年內）：2 項
- 中期（2-3年）：2 項
- 長期（3-5年）：1 項

輸出規則：只輸出這 5 行，不要標題、說明、空行或 Markdown 符號；
每行格式為「[G][期程] 行動：理由」，期程只能是「短期」「中期」「長期」。
"""
resp = completion(
    model="ollama/gemma4-plan:latest",
    messages=[{"role": "system", "content": "你是一位專業的 ESG 永續顧問。"},
              {"role": "user", "content": prompt}],
    max_tokens=4096,
    extra_body={"think": False},      # 嘗試一
    api_base="http://localhost:11434",
)
msg = resp["choices"][0]["message"]
content = msg.get("content") if hasattr(msg, "get") else msg.content
reasoning = getattr(msg, "reasoning_content", None)
print("usage:", resp["usage"])
print("content 長度:", len(content or ""))
print("content 開頭:", repr((content or "")[:400]))
print("reasoning 長度:", len(reasoning or ""))