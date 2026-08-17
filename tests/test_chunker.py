from chatbot.rag.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_short_text_returns_single_chunk():
    text = "Hello world."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert chunks == [text]


def test_long_text_is_split_into_multiple_chunks():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 220 for c in chunks)  # allow boundary slack


def test_chunks_cover_the_full_text_with_overlap():
    text = "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=15)
    joined = "".join(chunks)
    assert "A" * 50 in joined
    assert "C" * 50 in joined


def test_overlap_larger_than_chunk_size_is_clamped():
    text = "x" * 300
    # Should not raise or infinite-loop.
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=999)
    assert len(chunks) > 0
