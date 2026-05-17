"""Tests du cache d'embeddings et du StubEncoder."""

import numpy as np
import pytest

from muses.schemas.row import Row
from muses.schemas.tags import AxialTags
from muses.tables.embeddings import (
    EmbeddingsCache,
    StubEncoder,
    rebuild_for_table,
)
from muses.tables.jsonl_io import append_row


class TestStubEncoder:
    def test_dim(self):
        enc = StubEncoder(dim=8)
        out = enc.encode(["hello"])
        assert out.shape == (1, 8)
        assert out.dtype == np.float32

    def test_deterministic(self):
        enc = StubEncoder(dim=16)
        a = enc.encode(["même texte"])
        b = enc.encode(["même texte"])
        assert np.array_equal(a, b)

    def test_different_inputs_differ(self):
        enc = StubEncoder(dim=16)
        a = enc.encode(["alpha"])
        b = enc.encode(["beta"])
        assert not np.array_equal(a, b)

    def test_batch(self):
        enc = StubEncoder(dim=4)
        out = enc.encode(["a", "b", "c"])
        assert out.shape == (3, 4)


class TestEmbeddingsCache:
    def test_save_and_load(self, tmp_path):
        cache = EmbeddingsCache(tmp_path / "x.npy")
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        cache.save(arr)
        loaded = cache.load()
        assert np.array_equal(arr, loaded)

    def test_append_to_empty(self, tmp_path):
        cache = EmbeddingsCache(tmp_path / "x.npy")
        cache.append(np.array([[1.0]], dtype=np.float32))
        assert cache.load().shape == (1, 1)

    def test_append_extends(self, tmp_path):
        cache = EmbeddingsCache(tmp_path / "x.npy")
        cache.save(np.array([[1.0], [2.0]], dtype=np.float32))
        cache.append(np.array([[3.0]], dtype=np.float32))
        assert cache.load().shape == (3, 1)
        assert cache.load()[2, 0] == 3.0

    def test_append_rejects_dim_mismatch(self, tmp_path):
        cache = EmbeddingsCache(tmp_path / "x.npy")
        cache.save(np.zeros((1, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cache.append(np.zeros((1, 8), dtype=np.float32))


def test_rebuild_for_table_empty(tmp_path):
    jsonl = tmp_path / "t.jsonl"
    npy = tmp_path / "t.npy"
    n = rebuild_for_table(jsonl, npy, StubEncoder(dim=4))
    assert n == 0
    assert EmbeddingsCache(npy).load().shape == (0, 4)


def test_rebuild_for_table_aligns_with_jsonl_order(tmp_path):
    jsonl = tmp_path / "t.jsonl"
    npy = tmp_path / "t.npy"
    texts = ["alpha", "beta", "gamma"]
    for t in texts:
        append_row(
            jsonl,
            Row(
                level="fragment",
                tags=AxialTags(),
                content={"text": t},
                source="bootstrap",
            ),
        )
    n = rebuild_for_table(jsonl, npy, StubEncoder(dim=8))
    assert n == 3
    embeddings = EmbeddingsCache(npy).load()
    assert embeddings.shape == (3, 8)
    # On vérifie que les embeddings sont distincts et stables
    assert not np.array_equal(embeddings[0], embeddings[1])
