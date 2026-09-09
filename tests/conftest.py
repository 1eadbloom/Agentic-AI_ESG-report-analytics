"""pytest 共用工具：離線可跑的假 embedding 與 section 工廠。"""
import math
import zlib
from types import SimpleNamespace

import pytest


class FakeEmbedding:
    """確定性 character-frequency embedding（不用 sentence-transformers）。

    兩段文字共享的字元越多，餘弦相似度越高；用 zlib.crc32 保證跨 process
    穩定（不依賴 PYTHONHASHSEED）。
    """
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for ch in str(text).lower():
            v[zlib.crc32(ch.encode("utf-8")) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def __call__(self, inputs: list) -> list[list[float]]:
        return [self._vec(t) for t in inputs]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_section(**kw) -> SimpleNamespace:
    base = dict(
        section_id="sample::1", title="測試段落", text="範例文字內容範例文字內容",
        pillar="E", source_file="r.pdf", page_range="p1-3",
        company="示範公司", report_year="2024", doc_type="report",
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def patch_embedding(monkeypatch, fake_embedding):
    import citation
    monkeypatch.setattr(citation, "get_embedding_fn", lambda: fake_embedding)
    monkeypatch.setattr("interpretation_agent.get_embedding_fn", lambda: fake_embedding)
    yield fake_embedding