
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from docmind.chunker import TextChunk


DEFAULT_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class TextEncoder(Protocol):
    """Интерфейс модели для преобразования текста в векторы."""

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> NDArray[np.float32]:
        ...


def embed_chunks(
    chunks: list[TextChunk],
    encoder: TextEncoder,
    batch_size: int = 16,
) -> NDArray[np.float32]:
    """Преобразует текстовые фрагменты в матрицу эмбеддингов."""

    # Проверяем размер пакета обрабатываемых текстов
    if batch_size < 1:
        raise ValueError("batch_size должен быть больше нуля")

    # Не передаём модели пустой список
    if not chunks:
        raise ValueError("Список фрагментов пуст")

    # Извлекаем только текст, сохраняя исходный порядок фрагментов
    texts = [chunk["text"] for chunk in chunks]

    # Проверяем отсутствие пустых текстовых фрагментов
    if any(not text.strip() for text in texts):
        raise ValueError("Все фрагменты должны содержать текст")

    # Преобразуем тексты в нормализованные числовые векторы
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Приводим результат к двумерному массиву NumPy
    embeddings = np.asarray(vectors, dtype=np.float32)

    # Проверяем соответствие количества векторов и фрагментов
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Модель вернула некорректное количество эмбеддингов"
        )

    return embeddings
