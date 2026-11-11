
import numpy as np
from numpy.typing import NDArray

from docmind.chunker import TextChunk
from docmind.embeddings import TextEncoder


class SearchResult(TextChunk):
    """Результат поиска с оценкой смыслового сходства."""

    score: float


def search_chunks(
    query: str,
    chunks: list[TextChunk],
    embeddings: NDArray[np.float32],
    encoder: TextEncoder,
    top_k: int = 5,
) -> list[SearchResult]:
    """Находит наиболее близкие к запросу текстовые фрагменты."""

    # Проверяем корректность пользовательского запроса
    if not query.strip():
        raise ValueError("Поисковый запрос не должен быть пустым")

    if top_k < 1:
        raise ValueError("top_k должен быть больше нуля")

    # Если фрагментов нет, искать нечего
    if not chunks:
        return []

    document_vectors = np.asarray(embeddings, dtype=np.float32)

    # Проверяем соответствие матрицы исходным фрагментам
    if (
        document_vectors.ndim != 2
        or document_vectors.shape[0] != len(chunks)
        or document_vectors.shape[1] == 0
    ):
        raise ValueError(
            "Размерность эмбеддингов не соответствует фрагментам"
        )

    # Преобразуем поисковый запрос в вектор
    query_vectors = np.asarray(
        encoder.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )

    # Проверяем размерность вектора запроса
    if query_vectors.shape != (1, document_vectors.shape[1]):
        raise ValueError(
            "Размерность вектора запроса не совпадает с документом"
        )

    query_vector = query_vectors[0]

    # Вычисляем длины векторов
    document_norms = np.linalg.norm(document_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)

    # Проверяем отсутствие нулевых и некорректных векторов
    if (
        not np.all(np.isfinite(document_vectors))
        or not np.all(np.isfinite(query_vector))
        or np.any(document_norms == 0)
        or query_norm == 0
    ):
        raise ValueError(
            "Эмбеддинги содержат некорректные или нулевые значения"
        )

    # Вычисляем косинусное сходство запроса со всеми фрагментами
    scores = (
        document_vectors @ query_vector
    ) / (document_norms * query_norm)

    # Сортируем индексы по убыванию оценки сходства
    best_indices = np.argsort(-scores, kind="stable")[:top_k]

    # Формируем результаты, сохраняя исходные метаданные
    results: list[SearchResult] = [
        {
            **chunks[int(index)],
            "score": float(scores[index]),
        }
        for index in best_indices
    ]

    return results
