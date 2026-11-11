
import argparse

from docmind.pdf_reader import extract_pages
from docmind.chunker import chunk_pages


def main() -> None:
    """Запускает обработку PDF и выводит текстовые фрагменты."""

    parser = argparse.ArgumentParser(
        description="DocMind — обработка PDF-документов"
    )

    parser.add_argument(
        "pdf_path",
        help="Путь к PDF-документу",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Максимальный размер фрагмента в символах",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Количество перекрывающихся символов",
    )

    args = parser.parse_args()

    # Выполняем извлечение текста и разбиение на фрагменты
    try:
        pages = extract_pages(args.pdf_path)

        chunks = chunk_pages(
            pages,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"\nОбработано страниц: {len(pages)}")
    print(f"Создано фрагментов: {len(chunks)}")

    # Показываем первые три фрагмента для проверки
    for chunk in chunks[:3]:
        preview = chunk["text"][:160].replace("\n", " ")

        print(f"\n--- Фрагмент №{chunk['chunk_id']} ---")
        print(f"Страница: {chunk['page']}")
        print(f"Позиции: {chunk['start']}–{chunk['end']}")
        print(f"Текст: {preview}")


if __name__ == "__main__":
    main()
