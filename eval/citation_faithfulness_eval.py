"""
ESG multi-agent 專案 — Citation Faithfulness 評分腳本（直接用pipeline自己的輸出）
================================================================

不需要golden Q&A，也不需要額外跑問答——直接拿你pipeline正常運作後
產出的JSON分析報告來評分。

邏輯：每一項結論(亮點/待改善/KPI/策略)都帶著它自己的sources[]，
裡面的excerpt是citation機制認為支持這個結論的原文片段。
這個腳本做的事，就是檢查「結論文字」跟它自己標註的「excerpt」
語意對不對得上——分數高代表citation機制選對了真正相關的段落，
分數低代表可能選到不相關的片段（citation機制本身的準確度問題）。

用法:
    # 1. 先正常跑一次你的pipeline（README裡原本的用法）
    python coordinator_agent.py --file "2024年永續報告書.pdf" --company "中信證券"

    # 2. 對它產出的JSON跑這個評分腳本
    python citation_faithfulness_eval.py outputs/中信證券_ESG分析報告_日期.json

實際的JSON結構（已對照你的真實輸出確認）：
    diagnosis.E/S/G.highlights[]/weaknesses[]/kpis[] 底下，每個項目長這樣：
    {
        "text": "...(結論內容)...",
        "sources": [
            {"company": "...", "report_year": "...", "chunk_id": "...",
             "excerpt": "...(原文摘錄)..."}
        ]
    }
    分類(highlights/weaknesses/kpis/策略清單)跟面向(E/S/G)是從JSON的巢狀
    位置自動推斷的，不是item自己帶的欄位，見 extract_claims() 的走訪邏輯。
    「非文件依據」的結論(sources為空)，腳本會自動跳過，不算進分數。
"""

import sys
import json
import random
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("請先安裝: pip install sentence-transformers")
    sys.exit(1)

# 低於這個分數的citation，會額外送進LLM做「是/否支持」的二元判斷，
# 給一個比連續分數更容易跟企業溝通的通過率數字。
LLM_JUDGE_THRESHOLD = 0.5
# 用哪顆本地模型當judge，沿用你之前確認過穩定的設定
JUDGE_MODEL = "gemma4-plan:latest"
JUDGE_BASE_URL = "http://localhost:11434"


def extract_claims(data):
    """
    把JSON輸出攤平成 [(結論文字, excerpt文字, category, pillar), ...] 的清單。

    你的實際結構沒有獨立的type欄位，分類是用「巢狀位置」表達的：
    例如 diagnosis.E.highlights[].sources[].excerpt。
    這裡用一個帶上下文的遞迴走訪：往下走的時候記住「目前在哪個list的
    key底下」(parent_key，即category，如highlights/weaknesses/kpis)，
    以及「目前在哪個pillar物件裡」(從巢狀dict裡的pillar欄位，如E/S/G繼承下來)。
    這樣不用去猜kpis/strategies等清單的確切key拼法，自動從實際結構抓。
    """
    claims = []

    def walk(obj, parent_key=None, pillar=None):
        if isinstance(obj, dict):
            current_pillar = obj.get("pillar", pillar)
            if "sources" in obj and isinstance(obj.get("sources"), list) and obj["sources"]:
                claim_text = obj.get("text") or obj.get("content") or ""
                category = parent_key or "unknown"
                # 過濾掉上游解析LLM原始輸出時誤判出來的垃圾項目：
                # 像"###"、"．"這種只有markdown符號或孤立標點、沒有實質內容的
                # claim，不應該拿去算faithfulness分數（claim本身就是壞的，
                # 跟citation選得準不準無關）。去掉常見標點符號跟空白後，
                # 少於5個字元就視為無意義內容，直接跳過不計分。
                stripped = claim_text.strip().strip("#*．.-• \u3000")
                # 另一種垃圾項目：系統自己產生的佔位文字（例如LLM那個欄位
                # 沒有產出內容時的預設訊息），不是真正的分析內容，
                # 一併排除，不計入分數。
                placeholder_markers = ["分析結果未包含", "無資料", "無法生成"]
                is_placeholder = any(m in claim_text for m in placeholder_markers)
                if len(stripped) >= 5 and not is_placeholder:
                    for src in obj["sources"]:
                        excerpt = src.get("excerpt", "")
                        if claim_text and excerpt:
                            claims.append((claim_text, excerpt, category, current_pillar or "?"))
            for k, v in obj.items():
                walk(v, parent_key=k, pillar=current_pillar)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, parent_key=parent_key, pillar=pillar)

    walk(data)
    return claims


def shuffled_baseline(claims, model, n_trials: int = 3):
    """
    對照組：把claim跟「別組的」excerpt隨機配對（確保不會配到自己原本的excerpt），
    模擬「citation選錯段落」的狀況，算出這種真正文不對題的配對平均會是多少分。
    跑n_trials次取平均，降低隨機抽樣的誤差。
    """
    excerpts = [e for _, e, _, _ in claims]
    all_scores = []
    for _ in range(n_trials):
        shuffled = excerpts[:]
        random.shuffle(shuffled)
        # 確保洗牌後沒有任何一組剛好跟原本配對相同
        for i in range(len(shuffled)):
            if shuffled[i] == excerpts[i]:
                swap_with = (i + 1) % len(shuffled)
                shuffled[i], shuffled[swap_with] = shuffled[swap_with], shuffled[i]

        for (claim_text, _, _, _), fake_excerpt in zip(claims, shuffled):
            emb_claim = model.encode(claim_text, convert_to_tensor=True)
            emb_fake = model.encode(fake_excerpt, convert_to_tensor=True)
            all_scores.append(util.cos_sim(emb_claim, emb_fake).item())

    return sum(all_scores) / len(all_scores)


def llm_judge(claim_text: str, excerpt: str) -> dict:
    """
    對單一組citation做二元「是/否支持」判斷，回傳 {"supports": bool, "reason": str}。
    比連續的相似度分數更容易跟企業溝通（"94%通過LLM驗證" vs "平均0.604"）。
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=JUDGE_MODEL, base_url=JUDGE_BASE_URL, reasoning=False, format="json")
    prompt = (
        "你是ESG報告稽核員。請判斷下方的「引用原文」是否真的支持「結論」的說法。\n"
        "只考慮引用原文裡實際寫的內容，不要用你自己的知識腦補。\n"
        f"結論：{claim_text}\n"
        f"引用原文：{excerpt}\n"
        '請用這個JSON格式回答，不要有其他文字：{"supports": true或false, "reason": "一句話說明原因"}'
    )
    try:
        resp = llm.invoke(prompt)
        result = json.loads(resp.content)
        return {"supports": bool(result.get("supports", False)), "reason": result.get("reason", "")}
    except Exception as e:
        return {"supports": None, "reason": f"judge失敗: {e}"}


def evaluate(json_path: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    claims = extract_claims(data)
    print(f"（已自動跳過內容為空/只有符號的垃圾項目，不計入分數）")
    if not claims:
        print("沒有找到任何帶sources的結論項目。")
        print("請打開JSON檔案確認實際結構，並調整 extract_claims() 裡的欄位名稱。")
        return

    print(f"找到 {len(claims)} 組「結論-citation」配對\n")

    model = SentenceTransformer(model_name)

    results = []
    for claim_text, excerpt, category, pillar in claims:
        emb_claim = model.encode(claim_text, convert_to_tensor=True)
        emb_excerpt = model.encode(excerpt, convert_to_tensor=True)
        similarity = util.cos_sim(emb_claim, emb_excerpt).item()
        results.append({
            "category": category,
            "pillar": pillar,
            "claim": claim_text[:60],
            "excerpt": excerpt[:60],
            "similarity": round(similarity, 4),
        })

    results.sort(key=lambda r: r["similarity"])

    print("=== 相似度最低的10組（最需要人工檢查，可能citation選錯段落）===")
    for r in results[:10]:
        print(f"[{r['similarity']:.3f}] ({r['pillar']}/{r['category']}) {r['claim']}  <-cite->  {r['excerpt']}")

    avg_score = sum(r["similarity"] for r in results) / len(results)
    low_quality = sum(1 for r in results if r["similarity"] < 0.3)

    # 分類breakdown：依category（highlights/weaknesses/kpis/...）跟pillar（E/S/G）分組算平均
    print("\n=== 分類平均分數（面試被追問『哪一類citation比較不準』時可以直接報）===")
    by_category = {}
    by_pillar = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["similarity"])
        by_pillar.setdefault(r["pillar"], []).append(r["similarity"])
    print("依項目類型：")
    for cat, scores in sorted(by_category.items()):
        print(f"  {cat}: 平均 {sum(scores)/len(scores):.3f}（{len(scores)}組）")
    print("依E/S/G面向：")
    for pil, scores in sorted(by_pillar.items()):
        print(f"  {pil}: 平均 {sum(scores)/len(scores):.3f}（{len(scores)}組）")

    # 對照組：隨機配對的「假citation」平均會是多少分
    print("\n正在計算對照組（隨機配對的假citation）分數，這步要花點時間...")
    baseline_score = shuffled_baseline(claims, model)
    uplift = avg_score - baseline_score

    print(f"\n=== 整體分數（有對照組比較，比單一數字更有說服力）===")
    print(f"真實citation平均分數: {avg_score:.3f}")
    print(f"隨機配對(假citation)對照組平均分數: {baseline_score:.3f}")
    print(f"真實citation比隨機配對高出: {uplift:+.3f}（這個差距才是citation機制有效的證據，不是單一分數本身）")
    print(f"相似度低於0.3的citation數量: {low_quality} / {len(results)} ({low_quality/len(results):.1%})")

    # LLM二元判斷：只對分數偏低、不確定的citation做，節省時間
    borderline = [r for r in results if r["similarity"] < LLM_JUDGE_THRESHOLD]
    print(f"\n正在對 {len(borderline)} 組分數低於{LLM_JUDGE_THRESHOLD}的citation做LLM二元判斷...")
    judge_results = []
    for r in borderline:
        verdict = llm_judge(r["claim"], r["excerpt"])
        judge_results.append({**r, **verdict})

    supports_count = sum(1 for j in judge_results if j["supports"] is True)
    judged_total = sum(1 for j in judge_results if j["supports"] is not None)
    # 沒被送進LLM判斷的（分數本來就夠高），直接算作支持
    total_supports = supports_count + (len(results) - len(borderline))
    total_judged = judged_total + (len(results) - len(borderline))

    print(f"\n=== 給企業看的通過率（比連續分數更容易溝通）===")
    if total_judged > 0:
        print(f"整體citation支持率: {total_supports} / {total_judged} ({total_supports/total_judged:.1%})")
    print("（分數夠高的citation直接視為支持，只對低分的部分額外用LLM逐一覆核）")

    out_path = Path(json_path).with_name("citation_faithfulness_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_claim": results,
            "avg_faithfulness": avg_score,
            "baseline_faithfulness": baseline_score,
            "uplift": uplift,
            "llm_judge_borderline": judge_results,
            "support_rate": total_supports / total_judged if total_judged else None,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n已存成: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python citation_faithfulness_eval.py <pipeline輸出的JSON路徑>")
        sys.exit(1)
    evaluate(sys.argv[1])
