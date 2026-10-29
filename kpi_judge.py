"""
kpi_judge.py — 選用的「LLM 檢查員」：判斷原文是否明確寫出「這個指標＝這個數值」

與產生 KPI 的呼叫分開（可以用不同的模型），並且不直接相信模型的 yes/no：
模型必須「逐字引用」原文，程式再檢查引文真的出現在證據裡、而且含有該數值。
引文驗證不過，一律視為不支持。

啟用（.env）：
  ESG_KPI_JUDGE=1
  ESG_JUDGE_MODEL=openai/<模型名稱>            # LiteLLM 的模型字串
  ESG_JUDGE_BASE_URL=http://localhost:xxxx/v1   # 檢查員模型的 OpenAI 相容端點
  ESG_JUDGE_API_KEY=任意字串（若伺服器需要）
  ESG_JUDGE_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}   # 選用：關閉思考
"""
from __future__ import annotations
import json
import os
import re

from citation import _norm_for_numbers, value_tokens

MAX_EVIDENCE_CHARS = 2500

_PROMPT = """下面是一段永續報告書的原文，以及一個從中抽出的指標。
請判斷：原文是否「明確」寫出這個指標等於這個數值。

判斷規則：
- 數字必須與指標名稱屬於同一句話、或同一個表格的對應列與欄；只是數字相同、但屬於別的項目，不算支持。
- 若不確定，請回答 false。

指標名稱：{name}
數值：{value}

原文：
{evidence}

只輸出 JSON，不要任何其他文字：
{{"supported": true 或 false, "quote": "原文中支持的那句話，逐字複製；表格可用「…」連接多個片段；不支持就留空"}}
"""


def enabled() -> bool:
    return os.environ.get("ESG_KPI_JUDGE", "0") == "1" and bool(os.environ.get("ESG_JUDGE_MODEL"))


def _call_llm(prompt: str) -> str:
    from litellm import completion
    if not os.environ.get("ESG_JUDGE_MODEL"):
        raise RuntimeError("未設定 ESG_JUDGE_MODEL（請在 .env 加入，並確認是在專案根目錄執行）")
    kwargs: dict = {
        "model": os.environ["ESG_JUDGE_MODEL"],
        "messages": [
            {"role": "system", "content": "你是嚴謹的事實查核員，只輸出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(os.environ.get("ESG_JUDGE_MAX_TOKENS", "1024")),
        "temperature": 0,
    }
    if os.environ.get("ESG_JUDGE_BASE_URL"):
        kwargs["api_base"] = os.environ["ESG_JUDGE_BASE_URL"]
    if os.environ.get("ESG_JUDGE_API_KEY"):
        kwargs["api_key"] = os.environ["ESG_JUDGE_API_KEY"]
    if os.environ.get("ESG_JUDGE_EXTRA_BODY"):
        kwargs["extra_body"] = json.loads(os.environ["ESG_JUDGE_EXTRA_BODY"])
    resp = completion(**kwargs)
    return (resp["choices"][0]["message"]["content"] or "").strip()


def _parse(raw: str) -> dict | None:
    # 思考型模型可能先輸出 <think>…</think>，裡面的大括號不能被當成答案
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def _squash(t: str) -> str:
    return re.sub(r"\s+", "", _norm_for_numbers(t or ""))


def verify_quote(quote: str, evidence: str, value: str) -> bool:
    """引文每個片段都必須逐字出現在證據裡，且至少一個片段含有該數值。"""
    frags = [f for f in re.split(r"…|\.\.\.|｜", quote or "") if len(_squash(f)) >= 2]
    if not frags:
        return False
    ev = _squash(evidence)
    if not all(_squash(f) in ev for f in frags):
        return False
    toks = value_tokens(value)
    if not toks:
        return True
    return any(re.search(rf"(?<![\d.]){re.escape(toks[0])}(?!\d)", _squash(f)) for f in frags)


def judge_kpi(name: str, value: str, evidence_texts: list[str], call=None) -> bool | None:
    """回傳 True（有明確依據）、False（不支持）、None（模型沒有給出可解析的答案）。"""
    evidence = "\n---\n".join(evidence_texts)[:MAX_EVIDENCE_CHARS]
    raw = (call or _call_llm)(_PROMPT.format(name=name, value=value, evidence=evidence))
    d = _parse(raw)
    if d is None or "supported" not in d:
        return None
    if d.get("supported") is not True:
        return False
    return verify_quote(str(d.get("quote", "")), evidence, value)
