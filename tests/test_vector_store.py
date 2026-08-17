import numpy as np
import pytest

from chatbot.rag.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "index")


def _unit(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_empty_store_search_returns_nothing(store):
    assert store.is_empty()
    assert store.search(_unit([1, 0]), top_k=3, min_similarity=0.0) == []


def test_build_and_search_returns_closest_match(store):
    vectors = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])])
    store.build(vectors, texts=["a", "b", "c"], sources=["s1", "s1", "s2"])

    results = store.search(_unit([1, 0]), top_k=1, min_similarity=0.0)
    assert len(results) == 1
    assert results[0].text == "a"
    assert results[0].source == "s1"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_min_similarity_filters_out_weak_matches(store):
    vectors = np.stack([_unit([1, 0]), _unit([-1, 0])])
    store.build(vectors, texts=["match", "opposite"], sources=["s1", "s1"])

    results = store.search(_unit([1, 0]), top_k=2, min_similarity=0.5)
    assert len(results) == 1
    assert results[0].text == "match"


def test_persistence_round_trip(tmp_path):
    index_dir = tmp_path / "index"
    store1 = VectorStore(index_dir)
    vectors = np.stack([_unit([1, 0]), _unit([0, 1])])
    store1.build(vectors, texts=["a", "b"], sources=["s1", "s1"])

    store2 = VectorStore(index_dir)
    assert not store2.is_empty()
    assert store2.size == 2
    results = store2.search(_unit([1, 0]), top_k=1, min_similarity=0.0)
    assert results[0].text == "a"
