"""
inspect_pdf_page.py — 診斷：PDF 的數字在哪一步消失？

對含有指定關鍵字的頁面，印出三種版本：
  1. pdfplumber 原始文字
  2. 舊版 esg_utils.clean_text 之後（會刪除「只有數字的獨立一行」）
  3. 新版 search_agent._clean_page 之後
並顯示 pdfplumber 在該頁偵測到幾個表格。

用法：
  python inspect_pdf_page.py "2024年永續報告書(中)中信證券000616.pdf" "占整體投資金額"
  python inspect_pdf_page.py "<pdf 檔名>" "<關鍵字>" --max 2     # 最多看 2 頁
"""
import argparse
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")   # 避免 Windows cp950 編碼問題

import pdfplumber

from esg_utils import REPORTS_DIR, clean_text
from search_agent import _clean_page


def numeric_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if re.fullmatch(r"[\d.,%\s]+", ln) and ln.strip()]


ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("keyword")
ap.add_argument("--max", type=int, default=1)
args = ap.parse_args()

path = REPORTS_DIR / args.file
shown = 0
with pdfplumber.open(str(path)) as pdf:
    for i, page in enumerate(pdf.pages, 1):
        raw = page.extract_text() or ""
        if args.keyword not in raw:
            continue
        old = clean_text(raw)
        new = _clean_page(raw, i)
        try:
            n_tables = len(page.extract_tables())
        except Exception:
            n_tables = -1
        print("=" * 70)
        print(f"第 {i} 頁｜偵測到表格數：{n_tables}")
        print(f"純數字行數量：原始 {len(numeric_lines(raw))}｜舊版清理後 {len(numeric_lines(old))}"
              f"｜新版清理後 {len(numeric_lines(new))}")
        print("-" * 30, "原始文字（前 1200 字）", "-" * 30)
        print(raw[:1200])
        print("-" * 30, "舊版清理後（前 1200 字）", "-" * 30)
        print(old[:1200])
        print("-" * 30, "新版清理後（前 1200 字）", "-" * 30)
        print(new[:1200])
        shown += 1
        if shown >= args.max:
            break
if shown == 0:
    print("找不到含該關鍵字的頁面（請換個關鍵字，或確認檔名）")
