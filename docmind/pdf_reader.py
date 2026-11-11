from pathlib import Path
from typing import TypedDict

import pymupdf


class PageData(TypedDict):
    """Структура данных одной страницы документа."""

    page: int
    text: str


def extract_pages(pdf_path: str | Path) -> list[PageData]:
    """
    Извлекает текст из PDF-документа.

    Для каждой страницы возвращает её номер
    и извлечённый текст.
    """

    path = Path(pdf_path)

    # Проверяем, существует ли указанный файл
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    pages: list[PageData] = []

    # Открываем PDF и гарантируем закрытие после обработки
    with pymupdf.open(str(path)) as document:

        # Перебираем страницы с нумерацией начиная с единицы
        for page_number, page in enumerate(document, start=1):

            # Извлекаем текст и удаляем пробелы по краям
            text = page.get_text("text", sort=True).strip()

            # Сохраняем номер страницы вместе с текстом
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages