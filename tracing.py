"""
tracing.py — Langfuse 追蹤的薄包裝層

- 匯入時自行載入 .env，避免「先匯入、後載入環境變數」造成追蹤被關閉
- 未設定 LANGFUSE_PUBLIC_KEY 時，observe / get_client / propagate_attributes
  皆為無作用版本，離線測試與一般執行不受影響
"""
import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))

if ENABLED:
    from langfuse import observe, get_client, propagate_attributes
else:
    def observe(*args, **kwargs):
        # 同時支援 @observe 與 @observe(...) 兩種寫法
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    class _Noop:
        def __getattr__(self, _):
            return lambda *a, **k: None

    def get_client():
        return _Noop()

    @contextmanager
    def propagate_attributes(**kwargs):
        yield
