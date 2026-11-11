
import argparse

from docmind.chunker import chunk_pages
from docmind.embeddings import DEFAULT_MODEL_NAME, embed_chunks
from docmind.pdf_reader import extract_pages


def main() -> None:
    """Запускает обработку PDF и создание эмбеддингов."""

    parser = argparse.ArgumentParser(
        description="DocMind — интеллектуальная обработка PDF"
    )

    parser.add_argument(
        "pdf_path",
        help="Путь к PDF-документу",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Максимальный размер фрагмента",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Количество перекрывающихся символов",
    )

    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Создать векторные представления фрагментов",
    )

    args = parser.parse_args()

    # Извлекаем и разбиваем текст документа
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

    # Выводим первые три фрагмента для проверки
    for chunk in chunks[:3]:
        preview = chunk["text"][:160].replace("\n", " ")

        print(f"\n--- Фрагмент №{chunk['chunk_id']} ---")
        print(f"Страница: {chunk['page']}")
        print(f"Позиции: {chunk['start']}–{chunk['end']}")
        print(f"Текст: {preview}")

    # Завершаем работу, если эмбеддинги не запрошены
    if not args.embeddings:
        return

    if not chunks:
        print("\nНет текста для создания эмбеддингов.")
        return

    # Загружаем модель только при необходимости
    from sentence_transformers import SentenceTransformer

    print("\nЗагружаем модель эмбеддингов...")

    encoder = SentenceTransformer(DEFAULT_MODEL_NAME)

    # Преобразуем текстовые фрагменты в векторы
    embeddings = embed_chunks(chunks, encoder)

    print("\nЭмбеддинги успешно созданы!")
    print(f"Размер матрицы: {embeddings.shape}")

    # Показываем первые пять компонентов первого вектора
    print(
        "Первые 5 значений:",
        embeddings[0][:5].round(4).tolist(),
    )


if __name__ == "__main__":
    main()
