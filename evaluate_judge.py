"""
evaluate_judge.py — 在接進流程之前，先量測「LLM 檢查員」到底準不準

用一份已核對過的報告 JSON 建立三種測資：
  1. 正例：報告中的 KPI（扣掉已知錯誤的），檢查員應該回「支持」
  2. 已知錯誤：用 --bad 指定名稱關鍵字（例：董事長室配置資源），應該回「不支持」
  3. 合成反例：把某個 KPI 的名稱，配上別的 KPI 的數值（證據仍是原本被引用的片段），應該回「不支持」

同時用「確定性的鄰近檢查」跑同一批測資，讓你看出 LLM 檢查員有沒有帶來額外價值。

事前設定（.env 或環境變數）：
  ESG_JUDGE_MODEL=openai/<模型名稱>
  ESG_JUDGE_BASE_URL=http://localhost:xxxx/v1
  ESG_JUDGE_API_KEY=任意字串（若伺服器需要）
  ESG_JUDGE_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}   # 選用

用法：
  python evaluate_judge.py "outputs\\<報告>.json" --pdf "data\\reports\\<報告>.pdf" --bad 董事長室配置資源
  python evaluate_judge.py "outputs\\<報告>.json" --pdf "..." --limit 10      # 先少量試跑
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # 避免 Windows cp950 編碼問題

from dotenv import load_dotenv

load_dotenv()

import kpi_judge
from citation import value_near_name_in_texts, value_tokens
from verify_report import load_chunks


def build_cases(d: dict, chunks: dict[str, str], bad_names: list[str]):
    base = []
    for p in "ESG":
        for k in d["diagnosis"][p]["kpis"]:
            if "推算" in k["value"]:
                continue
            ev = [chunks.get(s["chunk_id"].split("::")[-1], "") for s in k["sources"]]
            ev = [e for e in ev if e]
            if ev and value_tokens(k["value"]):
                base.append({"pillar": p, "name": k["name"], "value": k["value"], "evidence": ev})
    cases = []
    for c in base:
        is_bad = any(b in c["name"] for b in bad_names)
        cases.append({**c, "kind": "已知錯誤" if is_bad else "正例", "expected": False if is_bad else True})
    # 合成反例：名稱 i ＋ 數值 j（證據用 i 的）
    n = len(base)
    for i, c in enumerate(base):
        if any(b in c["name"] for b in bad_names):
            continue
        for step in (7, 3, 5, 1):
            other = base[(i + step) % n]
            if value_tokens(other["value"])[0] != value_tokens(c["value"])[0]:
                if value_near_name_in_texts(c["name"], other["value"], c["evidence"]) is not True:
                    cases.append({**c, "value": other["value"], "kind": "合成反例", "expected": False})
                    break
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--pdf", default=None, help="用 PDF 重新切分取得 chunk 全文（不需要 ChromaDB）")
    ap.add_argument("--bad", nargs="*", default=[], help="已知錯誤 KPI 的名稱關鍵字")
    ap.add_argument("--limit", type=int, default=0, help="最多評估幾筆（0 表示全部）")
    args = ap.parse_args()

    import os
    missing = [k for k in ("ESG_JUDGE_MODEL", "ESG_JUDGE_BASE_URL") if not os.environ.get(k)]
    if missing:
        sys.exit(f"缺少環境變數：{', '.join(missing)}\n"
                 f"請在專案根目錄的 .env 設定（現在的工作資料夾：{Path.cwd()}）。")
    print(f"檢查員模型：{os.environ['ESG_JUDGE_MODEL']}｜端點：{os.environ['ESG_JUDGE_BASE_URL']}")
    d = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    ids = {s["chunk_id"] for p in "ESG" for k in d["diagnosis"][p]["kpis"] for s in k["sources"]}
    chunks = load_chunks(ids, args.pdf)
    cases = build_cases(d, chunks, args.bad)
    if args.limit:
        # 限制筆數時要涵蓋三種測資：已知錯誤先放，其餘正例與合成反例交錯
        bad = [c for c in cases if c["kind"] == "已知錯誤"]
        pos = [c for c in cases if c["kind"] == "正例"]
        neg = [c for c in cases if c["kind"] == "合成反例"]
        mixed = [x for pair in zip(pos, neg) for x in pair]
        cases = (bad + mixed + pos[len(neg):] + neg[len(pos):])[: args.limit]
    print(f"測資：{len(cases)} 筆（正例 {sum(c['kind']=='正例' for c in cases)}、"
          f"已知錯誤 {sum(c['kind']=='已知錯誤' for c in cases)}、"
          f"合成反例 {sum(c['kind']=='合成反例' for c in cases)}）\n")

    rows, lat = [], []
    for i, c in enumerate(cases, 1):
        t0 = time.time()
        try:
            verdict = kpi_judge.judge_kpi(c["name"], c["value"], c["evidence"])
        except Exception as e:                      # 連線或模型錯誤：記錄後繼續
            verdict = f"ERR:{type(e).__name__}:{str(e)[:160]}"
        lat.append(time.time() - t0)
        det = value_near_name_in_texts(c["name"], c["value"], c["evidence"])
        rows.append((c, verdict, det))
        mark = "✓" if verdict is c["expected"] else ("？" if verdict is None else "✗")
        shown = verdict if not isinstance(verdict, str) else verdict[:90]
        print(f"[{i:02d}] {mark} {c['kind']}｜{c['name'][:14]}＝{c['value'][:12]}｜檢查員={shown}｜鄰近檢查={det}")
        if isinstance(verdict, str) and i == 1:
            print("      （第一筆就出錯：先排除連線或設定問題再繼續，不必等全部跑完）")

    def rate(kind, expected, getter):
        sub = [r for r in rows if r[0]["kind"] in kind]
        if not sub:
            return "—"
        ok = sum(1 for r in sub if getter(r) is expected)
        none = sum(1 for r in sub if getter(r) is None)
        return f"{ok}/{len(sub)}" + (f"（另有 {none} 筆無法判斷）" if none else "")

    llm = lambda r: r[1] if not isinstance(r[1], str) else "ERR"
    det = lambda r: r[2]
    errs = sum(1 for r in rows if isinstance(r[1], str))
    print("\n" + "=" * 64)
    print(f"{'':20s}{'LLM 檢查員':<24s}{'確定性鄰近檢查':<20s}")
    print(f"{'正例被接受':<20s}{rate(['正例'], True, llm):<24s}{rate(['正例'], True, det):<20s}")
    print(f"{'已知錯誤被拒絕':<18s}{rate(['已知錯誤'], False, llm):<24s}{rate(['已知錯誤'], False, det):<20s}")
    print(f"{'合成反例被拒絕':<18s}{rate(['合成反例'], False, llm):<24s}{rate(['合成反例'], False, det):<20s}")
    print(f"\n平均每筆耗時：{statistics.mean(lat):.1f} 秒｜呼叫錯誤：{errs} 筆")
    print("判讀：正例被接受率低，代表檢查員太嚴、會誤刪好 KPI；反例被拒絕率低，代表它抓不到錯配。\n"
          "兩者都要高，而且要比「確定性鄰近檢查」好，才值得多花這些時間。")


if __name__ == "__main__":
    main()
