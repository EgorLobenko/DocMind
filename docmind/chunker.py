
from typing import TypedDict

from docmind.pdf_reader import PageData


class TextChunk(TypedDict):
    """Структура одного текстового фрагмента."""

    chunk_id: int
    page: int
    start: int
    end: int
    text: str


def chunk_pages(
    pages: list[PageData],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    """Разбивает текст страниц на пересекающиеся фрагменты."""

    # Проверяем корректность настроек разбиения
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше нуля")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap должен быть неотрицательным и меньше chunk_size"
        )

    chunks: list[TextChunk] = []

    # Обрабатываем каждую страницу независимо
    for page in pages:
        text = page["text"]

        # Пропускаем страницы без содержательного текста
        if not text.strip():
            continue

        start = 0

        while start < len(text):
            # Определяем конец текущего фрагмента
            end = min(start + chunk_size, len(text))

            # Сохраняем текст и его положение в документе
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "page": page["page"],
                "start": start,
                "end": end,
                "text": text[start:end],
            })

            # Завершаем обработку, если достигли конца страницы
            if end == len(text):
                break

            # Смещаем начало с учётом перекрытия
            start = end - overlap

    return chunks
