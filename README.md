<div align="center">

# ◈ DOCMIND

### Search by meaning. Navigate by source.

*A learning-driven semantic search engine for PDF documents.*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-0F766E?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-NLP%20%2B%20Semantic%20Search-202938?style=for-the-badge)

</div>

---

## 01 / The idea

> **Your documents contain answers. DocMind helps you find where they live.**

**DocMind** is a Python project currently in development. Its goal is to make PDF documents searchable **by meaning**, not only by exact words.

Instead of reading a long document page by page, a user will enter a question or a search query and receive relevant text passages, together with their original page numbers.

The first version will focus on **retrieving source passages**, not generating AI-written answers.

## 02 / Planned features

- [ ] Upload a PDF document containing selectable text.
- [ ] Extract text while preserving page numbers.
- [ ] Split long text into manageable, overlapping chunks.
- [ ] Convert chunks and queries into vector embeddings.
- [ ] Rank passages by semantic similarity to a query.
- [ ] Display relevant excerpts and their source pages in a web interface.

## 03 / How it will work

**PDF → Text extraction → Chunking → Embeddings → Similarity search → Ranked passages + page references**

Semantic search compares vector representations of text. This makes it possible to find related passages even when the query and the document do not use exactly the same words.

## 04 / Tech blueprint

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| Core | Python 3.12 | Application logic |
| PDF processing | PyMuPDF | Extract text and page information |
| NLP | SentenceTransformers | Generate multilingual text embeddings |
| Numerical operations | NumPy | Compare vectors and calculate similarity |
| Web interface | Streamlit | Upload files and explore search results |
| Testing | pytest | Check the correctness of core functions |
| Version control | Git & GitHub | Track changes and develop features in branches |

*The stack is planned and may evolve as the project develops.*

## 05 / Roadmap

- [x] Initialize the local Git repository.
- [x] Create an isolated Python environment and configure `.gitignore`.
- [ ] Add the first project documentation and commit.
- [ ] Implement PDF text extraction.
- [ ] Implement text chunking.
- [ ] Generate text embeddings.
- [ ] Build a semantic search function.
- [ ] Create the Streamlit interface.
- [ ] Add automated tests and improve error handling.
- [ ] Document setup, publish the repository and release `v1.0`.

## 06 / Current status

**Stage:** Initial setup.

DocMind is **not yet a runnable application**. Installation and launch instructions will be added when the first working version is ready.

The initial release is intended for PDFs with an extractable text layer. OCR for scanned documents is outside the scope of `v1.0`.

## 07 / Beyond v1.0

Possible future improvements include searching across multiple documents, highlighting matching passages, and adding source-grounded question answering (RAG).

---

<div align="center">

**Find the passage. Understand the context. Keep the source.**

<sub>Built step by step — one meaningful commit at a time.</sub>

</div>
