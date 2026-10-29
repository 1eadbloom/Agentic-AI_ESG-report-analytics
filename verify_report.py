"""
verify_report.py — 檢查一份 ESG 分析報告（JSON）是否符合這一輪修正的預期

只讀 outputs/ 裡的 .json，不呼叫 LLM。需要「被引用 chunk 的全文」時：
  - 預設從 ChromaDB 依 chunk_id 取
  - 或用 --pdf 直接從 PDF 重新切分（不需要 ChromaDB，結果與索引一致）

用法：
  python verify_report.py                      # 檢查 outputs/ 裡最新的報告
  python verify_report.py "outputs\\中信證券_ESG分析報告_20261025_xxxx.json"
  python verify_report.py "<json>" --pdf "data\\reports\\2024年永續報告書(中)中信證券000616.pdf"
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # 避免 Windows cp950 編碼問題

import interpretation_agent as ia
from citation import contains_number, value_tokens

PLACEHOLDERS = ("（無法評估）", "（本份報告書中未找到相關內容）",
                "分析結果未包含改善建議", "（無資料）")
results: list[tuple[str, str, str]] = []     # (狀態, 檢查項目, 說明)


def check(ok: bool | None, name: str, detail: str = "", warn_only: bool = False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    if ok is None:
        status = "SKIP"
    results.append((status, name, detail))


# ── 取得被引用 chunk 的全文 ───────────────────────────────────────
def load_chunks(ids: set[str], pdf: str | None) -> dict[str, str]:
    """回傳 {chunk 編號（'::' 後面的部分）: 全文}。"""
    key = lambda cid: cid.split("::")[-1]
    if pdf:
        import search_agent as sa
        p = Path(pdf)
        secs = sa.split_into_sections(sa.extract_pdf(p), p.name, "", "")
        return {key(s.section_id): s.text for s in secs}
    from esg_utils import get_collection
    _, col = get_collection()
    got = col.get(ids=sorted(ids), include=["documents"])
    return {key(i): d for i, d in zip(got["ids"], got["documents"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default=None,
                    help="報告 JSON；省略時使用 outputs/ 裡最新的一份")
    ap.add_argument("--pdf", default=None, help="改用 PDF 重新切分來取得 chunk 全文")
    args = ap.parse_args()

    if args.json_path is None:
        from esg_utils import OUTPUTS_DIR
        found = sorted(OUTPUTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime)
        if not found:
            sys.exit(f"{OUTPUTS_DIR} 裡沒有 .json 報告，請先執行 coordinator_agent.py")
        args.json_path = str(found[-1])
    print(f"檢查檔案：{args.json_path}\n")
    d = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    dg, sol = d.get("diagnosis") or {}, d.get("solution") or {}

    # 流程本身失敗時（例如 LLM 服務崩潰），診斷或策略是空的：直接說明，不要丟出 KeyError
    empty = [p for p in "ESG" if not (dg.get(p) or {}).get("highlights")]
    if not dg or empty or not sol.get("strategies"):
        print("[FAIL] 這份報告的內容是空的，代表產生它的那次流程失敗了，不是檢查腳本的問題。")
        print(f"       診斷缺少的面向：{empty or '（整個診斷為空）'}｜策略數：{len(sol.get('strategies', []))}")
        errs = d.get("errors") or []
        print("       報告內記錄的錯誤（最重要的線索）：")
        for e in errs[:8]:
            print(f"         - {str(e)[:200]}")
        if not errs:
            print("         （沒有記錄）請看執行 coordinator_agent.py 時的終端機輸出。")
        print("       建議：先確認 Ollama 正常（ollama ps）、Qwen 沒有佔著顯存，再重跑流程。")
        sys.exit(1)

    # ── 1. 結構 ──────────────────────────────────────────────────
    for p in "ESG":
        pa = dg[p]
        real_h = [h for h in pa["highlights"] if not h["text"].startswith(PLACEHOLDERS)]
        real_w = [w for w in pa["weaknesses"] if not w["text"].startswith(PLACEHOLDERS)]
        check(bool(real_h) and bool(real_w), f"{p} 面向有實際的亮點與待改善",
              f"亮點 {len(real_h)}、待改善 {len(real_w)}")
    counts = {p: sum(1 for s in sol["strategies"] if s["pillar"] == p) for p in "ESG"}
    check(all(n >= 3 for n in counts.values()), "每個面向至少 3 條策略", str(counts))
    check(not d.get("errors"), "執行紀錄沒有警告或錯誤", "; ".join(d.get("errors", []))[:200])

    # KPI 數量：其他檢查都只看「有的東西對不對」，抓不到「該有的東西不見了」
    kcount = {p: len(dg[p]["kpis"]) for p in "ESG"}
    check(all(n >= 4 for n in kcount.values()), "每個面向至少 4 個 KPI（避免整批消失沒被發現）",
          str(kcount), warn_only=True)

    # ── 2. 殘渣 ──────────────────────────────────────────────────
    junk = []
    for p in "ESG":
        for k in dg[p]["kpis"]:
            if not ia._valid_kpi(ia._strip_md(k["name"]), re.sub(r"（推算.*?）", "", k["value"])):
                junk.append(f"{p}:{k['name'][:14]}={k['value'][:14]}")
        for item in dg[p]["highlights"] + dg[p]["weaknesses"]:
            if re.fullmatch(r"[#\-*_\s]*", item["text"]) or len(item["text"].strip()) < 8:
                junk.append(f"{p}:短條目「{item['text'][:10]}」")
    check(not junk, "沒有垃圾 KPI／標題殘渣", "；".join(junk[:6]))
    # 找出雜訊的「位置」（欄位路徑加上前後文），方便直接判斷是哪一句出問題
    noise_pat = re.compile(r"\$[^$]{1,30}\$|人人次|\ufffd")
    noise_hits = []

    def _walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            for m in noise_pat.finditer(node):
                ctx = node[max(0, m.start() - 15):m.end() + 15].replace("\n", " ")
                noise_hits.append(f"{path}：…{ctx}…")
    _walk(d, "")
    check(not noise_hits, "沒有 LaTeX／重複量詞／亂碼字元",
          "；".join(noise_hits[:3]), warn_only=True)

    # ── 3. 頁碼 ──────────────────────────────────────────────────
    pages = [s["page_range"] for p in "ESG"
             for it in dg[p]["highlights"] + dg[p]["weaknesses"] + dg[p]["kpis"]
             for s in it["sources"] if s["chunk_id"] != "unattributed"]
    wide = [pg for pg in pages if pg == "p1-106" or
            (re.fullmatch(r"p(\d+)-(\d+)", pg) and
             int(pg.split("-")[1]) - int(pg.split("p")[1].split("-")[0]) > 3)]
    check(pages and not wide, "引用頁碼是具體頁碼（不是整份文件）",
          f"{len(pages)} 筆引用，過寬 {len(wide)} 筆")

    # ── 4. KPI 數值是否出現在被引用的 chunk ──────────────────────
    ids = {s["chunk_id"] for p in "ESG" for k in dg[p]["kpis"] for s in k["sources"]
           if s["chunk_id"] != "unattributed"}
    try:
        chunks = load_chunks(ids, args.pdf)
    except Exception as e:                       # 沒有 ChromaDB 或 PDF 時略過這項
        chunks = {}
        check(None, "KPI 數值出現在被引用的 chunk", f"無法取得 chunk 全文：{e}")
    if chunks:
        total = hit = 0
        miss, flagged = [], 0
        for p in "ESG":
            for k in dg[p]["kpis"]:
                toks = value_tokens(re.sub(r"（推算.*?）", "", k["value"]))
                if not toks:
                    continue
                if "推算" in k["value"]:
                    flagged += 1
                    continue
                total += 1
                texts = [chunks.get(s["chunk_id"].split("::")[-1], "") for s in k["sources"]]
                if any(contains_number(t, toks[0]) for t in texts):
                    hit += 1
                else:
                    miss.append(f"{p}:{k['name'][:12]}={k['value'][:12]}")
        rate = hit / total if total else 1.0
        check(rate >= 0.9, "KPI 數值出現在被引用的 chunk（至少 90%）",
              f"{hit}/{total} = {rate:.0%}；未命中：{'；'.join(miss[:5])}；已標示推算：{flagged}")

    # ── 4b. 語意合理性：KPI 名稱的關鍵字是否出現在該數值附近 ──────
    # 只比對數字字串會漏掉「數字存在、但跟名稱無關」的 KPI（例：董事長室配置資源＝100%）。
    # 邏輯與流程內的過濾共用（citation.value_near_name_in_texts）。
    if chunks:
        from citation import value_near_name_in_texts
        weak = []
        for p in "ESG":
            for k in dg[p]["kpis"]:
                if "推算" in k["value"]:
                    continue
                texts = [chunks.get(s["chunk_id"].split("::")[-1], "") for s in k["sources"]]
                if value_near_name_in_texts(k["name"], re.sub(r"（推算.*?）", "", k["value"]),
                                            texts) is False:
                    weak.append(f"{p}:{k['name'][:16]}={k['value'][:12]}")
        check(not weak, "KPI 名稱的關鍵字出現在數值附近（語意合理性）",
              "請人工核對：" + "；".join(weak), warn_only=True)

    # ── 5. 最佳實踐與備援標示 ────────────────────────────────────
    bp = sol["best_practices"]
    check(all(b["suggestion"].strip() for b in bp), "最佳實踐沒有空白", f"{len(bp)} 筆")
    fb = [b["pillar"] for b in bp if "預設範本" in b.get("source", "")]
    check(not fb, "最佳實踐來自知識庫（不是預設範本）",
          f"使用範本的面向：{fb}；來源：{[b.get('source') for b in bp]}", warn_only=True)
    fb_s = sum(1 for s in sol["strategies"] if "預設範本" in s.get("rationale", ""))
    check(fb_s == 0, "策略不是預設範本", f"預設範本策略 {fb_s} 條", warn_only=True)

    # ── 輸出 ─────────────────────────────────────────────────────
    width = max(len(n) for _, n, _ in results)
    for st, name, detail in results:
        print(f"[{st}] {name.ljust(width)}  {detail}")
    fails = sum(1 for s, _, _ in results if s == "FAIL")
    warns = sum(1 for s, _, _ in results if s == "WARN")
    print(f"\n結果：{fails} 項 FAIL、{warns} 項 WARN。"
          f"（FAIL 代表這輪修正沒有生效或有退步；WARN 值得看一下但不一定是問題。）")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
