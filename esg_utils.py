"""
esg_utils.py — ESG Agent 共用工具
路徑常數、文字清理、ESG 分類關鍵字庫、LLM 呼叫封裝
"""
from __future__ import annotations
import os, re
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR  = PROJECT_ROOT / "data" / "reports"
KB_DIR       = PROJECT_ROOT / "data" / "knowledge_base"
PROCESSED_DIR= PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
CHROMA_DIR   = PROJECT_ROOT / "chroma_db"

for d in [REPORTS_DIR, KB_DIR, PROCESSED_DIR, OUTPUTS_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── ESG 關鍵字分類庫 ──────────────────────────────────────────────
ESG_KEYWORDS = {
    "E": [
        "碳排放", "溫室氣體", "碳中和", "淨零", "氣候變遷", "再生能源", "節能",
        "用電", "耗能", "水資源", "廢棄物", "環境", "綠色", "碳足跡", "生物多樣性",
        "永續採購", "綠色採購", "空氣品質", "减碳", "碳盤查", "carbon", "emission",
        "renewable", "energy", "waste", "water", "climate", "environment",
    ],
    "S": [
        "員工", "人才", "薪資", "福利", "職業安全", "勞工", "勞動", "人權",
        "多元共融", "教育訓練", "社會參與", "公益", "客戶", "消費者保護",
        "公平待客", "社區", "供應鏈", "女性", "職安衛", "身心障礙",
        "employee", "labor", "safety", "community", "customer", "diversity",
        "human rights", "training", "social",
    ],
    "G": [
        "治理", "董事會", "獨立董事", "審計委員會", "薪酬委員會", "風險管理",
        "內部控制", "法令遵循", "反貪腐", "誠信", "資訊安全", "稅務", "揭露",
        "股東", "利害關係人", "永續治理", "ESG委員會", "重大性",
        "governance", "board", "compliance", "audit", "risk", "transparency",
        "anti-corruption", "shareholder",
    ],
}

GRI_STANDARDS = {
    "E": ["GRI 302 能源", "GRI 303 水與廢水", "GRI 305 排放", "GRI 306 廢棄物"],
    "S": ["GRI 401 就業", "GRI 403 職業健康安全", "GRI 404 訓練與教育", "GRI 413 當地社區"],
    "G": ["GRI 205 反腐敗", "GRI 206 反競爭行為", "GRI 418 客戶隱私"],
}

# ── 文字清理 ──────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 移除 PDF CID 亂碼字符（如 (cid:12345)）
    text = re.sub(r"\(cid:\d+\)", "", text)
    # 移除頁首/頁尾樣板文字
    text = re.sub(r"2024 Sustainability Report\s*", "", text)
    text = re.sub(r"Sustainability Report\s*", "", text)
    # 移除獨立頁碼行
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # 移除只包含 CID 字符的行
    text = re.sub(r"(?m)^[\s\(\):\d]+$", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ── ESG 段落分類 ──────────────────────────────────────────────────
def classify_pillar(text: str) -> str:
    t = text.lower()
    scores = {p: sum(1 for k in kws if k.lower() in t)
              for p, kws in ESG_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "O"  # O = Other/General

# ── LLM 呼叫（LiteLLM，支援 Ollama）────────────────────────────────
def call_llm(prompt: str, system: str = "你是一位專業的 ESG 永續顧問。",
             model: str | None = None, max_tokens: int = 2048) -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    from litellm import completion
    m = model or os.environ.get("ESG_MODEL", "ollama/gemma3:4b")
    api_key  = os.environ.get("LITELLM_API_KEY")
    base_url = os.environ.get("LITELLM_BASE_URL")
    kwargs: dict = {
        "model": m,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": max_tokens,
    }
    if api_key:  kwargs["api_key"]  = api_key
    if m.startswith("ollama/") and not base_url:
        base_url = "http://localhost:11434"
    if base_url: kwargs["api_base"] = base_url
    resp = completion(**kwargs)
    return resp["choices"][0]["message"]["content"].strip()

# ── Embedding（sentence-transformers，與 HW3 共用邏輯）──────────────
class EmbeddingFn:
    _instance = None
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._model.encode(input, show_progress_bar=False,
                                  convert_to_numpy=True).tolist()
    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False,
                                  convert_to_numpy=True)[0].tolist()
    def name(self) -> str: return f"st-{self.model_name}"
    def get_config(self) -> dict: return {"model_name": self.model_name}
    @staticmethod
    def build_from_config(c): return EmbeddingFn(c.get("model_name"))

def get_embedding_fn() -> EmbeddingFn:
    if EmbeddingFn._instance is None:
        EmbeddingFn._instance = EmbeddingFn()
    return EmbeddingFn._instance

# ── ChromaDB collection ────────────────────────────────────────────
def get_collection(ef: EmbeddingFn | None = None):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = ef or get_embedding_fn()
    return client, client.get_or_create_collection(
        name="esg_reports",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
