
import pytest

from docmind.chunker import chunk_pages


def test_chunk_pages_overlap() -> None:
    """Проверяет правильность перекрытия фрагментов."""

    pages = [{"page": 1, "text": "abcdefghij"}]

    result = chunk_pages(pages, chunk_size=6, overlap=2)

    assert result == [
        {
            "chunk_id": 1,
            "page": 1,
            "start": 0,
            "end": 6,
            "text": "abcdef",
        },
        {
            "chunk_id": 2,
            "page": 1,
            "start": 4,
            "end": 10,
            "text": "efghij",
        },
    ]


def test_chunk_pages_do_not_mix_pages() -> None:
    """Проверяет раздельную обработку страниц."""

    pages = [
        {"page": 1, "text": "ABCDEFG"},
        {"page": 2, "text": "1234"},
    ]

    result = chunk_pages(pages, chunk_size=5, overlap=1)

    assert [
        (chunk["page"], chunk["text"])
        for chunk in result
    ] == [
        (1, "ABCDE"),
        (1, "EFG"),
        (2, "1234"),
    ]

    assert [chunk["chunk_id"] for chunk in result] == [1, 2, 3]
    assert result[2]["start"] == 0


def test_empty_pages_are_skipped() -> None:
    """Проверяет пропуск страниц без содержательного текста."""

    pages = [
        {"page": 1, "text": " "},
        {"page": 2, "text": "Hello"},
    ]

    result = chunk_pages(pages, chunk_size=10, overlap=2)

    assert len(result) == 1
    assert result[0]["page"] == 2


def test_exact_chunk_size() -> None:
    """Проверяет текст, равный размеру одного фрагмента."""

    pages = [{"page": 1, "text": "abcde"}]

    result = chunk_pages(pages, chunk_size=5, overlap=2)

    assert len(result) == 1
    assert result[0]["text"] == "abcde"


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_invalid_chunk_settings(
    chunk_size: int,
    overlap: int,
) -> None:
    """Проверяет отклонение некорректных настроек."""

    pages = [{"page": 1, "text": "Hello"}]

    with pytest.raises(ValueError):
        chunk_pages(
            pages,
            chunk_size=chunk_size,
            overlap=overlap,
        )
