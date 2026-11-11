
import numpy as np
import pytest

from docmind.search import search_chunks


class FakeEncoder:
    """Имитирует модель эмбеддингов для тестирования."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray([vector], dtype=np.float32)
        self.calls = 0

    def encode(
        self,
        texts: list[str],
        **settings: object,
    ) -> np.ndarray:
        """Возвращает заранее определённый вектор запроса."""

        self.calls += 1
        return self.vector


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Возвращает три тестовых текстовых фрагмента."""

    return [
        {
            "chunk_id": 1,
            "page": 1,
            "start": 0,
            "end": 5,
            "text": "Текст A",
        },
        {
            "chunk_id": 2,
            "page": 2,
            "start": 0,
            "end": 5,
            "text": "Текст B",
        },
        {
            "chunk_id": 3,
            "page": 3,
            "start": 0,
            "end": 5,
            "text": "Текст C",
        },
    ]


def test_search_ranks_chunks_by_similarity(sample_chunks) -> None:
    """Проверяет порядок результатов по косинусному сходству."""

    embeddings = np.array(
        [
            [0.0, 3.0],
            [2.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    encoder = FakeEncoder([1.0, 0.0])

    results = search_chunks(
        query="тестовый запрос",
        chunks=sample_chunks,
        embeddings=embeddings,
        encoder=encoder,
        top_k=2,
    )

    assert len(results) == 2
    assert [item["chunk_id"] for item in results] == [2, 3]
    assert results[0]["page"] == 2
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.70710678)
    assert encoder.calls == 1


def test_top_k_larger_than_chunk_count(sample_chunks) -> None:
    """Проверяет запрос большего числа результатов, чем имеется."""

    embeddings = np.eye(3, dtype=np.float32)
    encoder = FakeEncoder([1.0, 0.0, 0.0])

    results = search_chunks(
        "поиск",
        sample_chunks,
        embeddings,
        encoder,
        top_k=10,
    )

    assert len(results) == 3


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(query: str) -> None:
    """Проверяет запрет пустых поисковых запросов."""

    encoder = FakeEncoder([1.0, 0.0])

    with pytest.raises(ValueError, match="запрос"):
        search_chunks(
            query,
            [],
            np.empty((0, 2)),
            encoder,
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    """Проверяет некорректное количество результатов."""

    encoder = FakeEncoder([1.0, 0.0])

    with pytest.raises(ValueError, match="top_k"):
        search_chunks(
            "поиск",
            [],
            np.empty((0, 2)),
            encoder,
            top_k=top_k,
        )


def test_empty_chunks_return_empty_results() -> None:
    """Проверяет поиск по документу без фрагментов."""

    encoder = FakeEncoder([1.0, 0.0])

    results = search_chunks(
        "поиск",
        [],
        np.empty((0, 2)),
        encoder,
    )

    assert results == []
    assert encoder.calls == 0


def test_incorrect_document_shape_is_rejected(sample_chunks) -> None:
    """Проверяет соответствие количества эмбеддингов."""

    encoder = FakeEncoder([1.0, 0.0])
    embeddings = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Размерность"):
        search_chunks(
            "поиск",
            sample_chunks,
            embeddings,
            encoder,
        )


def test_query_dimension_mismatch_is_rejected(sample_chunks) -> None:
    """Проверяет несовпадение размерностей векторов."""

    encoder = FakeEncoder([1.0, 0.0, 0.0])
    embeddings = np.ones((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Размерность"):
        search_chunks(
            "поиск",
            sample_chunks,
            embeddings,
            encoder,
        )


def test_zero_vector_is_rejected(sample_chunks) -> None:
    """Проверяет отклонение нулевых эмбеддингов."""

    encoder = FakeEncoder([1.0, 0.0])
    embeddings = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="нулевые"):
        search_chunks(
            "поиск",
            sample_chunks,
            embeddings,
            encoder,
        )
