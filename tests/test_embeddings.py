
import numpy as np
import pytest

from docmind.embeddings import embed_chunks


class FakeEncoder:
    """Имитирует модель без скачивания её весов."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self.texts: list[str] | None = None
        self.settings: dict | None = None

    def encode(
        self,
        texts: list[str],
        **settings: object,
    ) -> np.ndarray:
        """Возвращает заранее подготовленные векторы."""

        self.texts = texts
        self.settings = settings

        return self.vectors


def test_embed_chunks_returns_matrix_in_original_order() -> None:
    """Проверяет порядок векторов и параметры модели."""

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "start": 0,
            "end": 6,
            "text": "Привет",
        },
        {
            "chunk_id": 2,
            "page": 2,
            "start": 0,
            "end": 3,
            "text": "Мир",
        },
    ]

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    encoder = FakeEncoder(expected)

    result = embed_chunks(chunks, encoder, batch_size=8)

    np.testing.assert_array_equal(result, expected)

    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    assert encoder.texts == ["Привет", "Мир"]

    assert encoder.settings == {
        "batch_size": 8,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }


def test_embed_chunks_rejects_empty_list() -> None:
    """Проверяет обработку пустого списка фрагментов."""

    encoder = FakeEncoder(np.empty((0, 3)))

    with pytest.raises(ValueError, match="пуст"):
        embed_chunks([], encoder)


def test_embed_chunks_rejects_blank_text() -> None:
    """Проверяет отклонение пустого текста."""

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "start": 0,
            "end": 3,
            "text": "   ",
        }
    ]

    encoder = FakeEncoder(np.zeros((1, 3)))

    with pytest.raises(ValueError, match="содержать текст"):
        embed_chunks(chunks, encoder)


@pytest.mark.parametrize("batch_size", [0, -2])
def test_embed_chunks_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    """Проверяет некорректный размер пакета."""

    encoder = FakeEncoder(np.empty((0, 3)))

    with pytest.raises(ValueError, match="batch_size"):
        embed_chunks([], encoder, batch_size=batch_size)


def test_embed_chunks_checks_output_shape() -> None:
    """Проверяет число векторов, возвращённых моделью."""

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "start": 0,
            "end": 4,
            "text": "text",
        },
        {
            "chunk_id": 2,
            "page": 1,
            "start": 4,
            "end": 8,
            "text": "more",
        },
    ]

    # Имитируем ошибку модели: один вектор для двух фрагментов
    encoder = FakeEncoder(np.array([[1, 0, 0]]))

    with pytest.raises(ValueError, match="количество эмбеддингов"):
        embed_chunks(chunks, encoder)
