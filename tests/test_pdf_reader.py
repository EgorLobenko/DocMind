
from pathlib import Path

import pymupdf
import pytest

from docmind.pdf_reader import extract_pages


def create_test_pdf(path: Path, texts: list[str]) -> None:
    """Создаёт временный PDF для автоматических тестов."""

    with pymupdf.open() as document:
        for text in texts:
            # Создаём отдельную страницу
            page = document.new_page()

            # Добавляем текст, если он не пустой
            if text:
                page.insert_text((72, 72), text)

        # Сохраняем созданный документ
        document.save(str(path))


def test_extract_pages_returns_text_and_numbers(tmp_path: Path) -> None:
    """Проверяет извлечение текста и нумерацию страниц."""

    pdf_path = tmp_path / "sample.pdf"

    create_test_pdf(
        pdf_path,
        [
            "Machine learning basics",
            "Neural networks introduction",
        ],
    )

    result = extract_pages(pdf_path)

    assert len(result) == 2

    assert result[0] == {
        "page": 1,
        "text": "Machine learning basics",
    }

    assert result[1] == {
        "page": 2,
        "text": "Neural networks introduction",
    }


def test_extract_pages_handles_empty_page(tmp_path: Path) -> None:
    """Проверяет обработку страницы без текста."""

    pdf_path = tmp_path / "empty_page.pdf"

    create_test_pdf(pdf_path, ["Hello", ""])

    result = extract_pages(pdf_path)

    assert len(result) == 2
    assert result[1]["page"] == 2
    assert result[1]["text"] == ""


def test_extract_pages_raises_error_for_missing_file(
    tmp_path: Path,
) -> None:
    """Проверяет ошибку при отсутствии документа."""

    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pages(missing_path)


def test_extract_pages_accepts_string_path(tmp_path: Path) -> None:
    """Проверяет поддержку строкового пути к файлу."""

    pdf_path = tmp_path / "document.pdf"

    create_test_pdf(pdf_path, ["DocMind test"])

    result = extract_pages(str(pdf_path))

    assert result[0]["text"] == "DocMind test"
