"""Vector store backend abstraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Protocol

from langchain_community.vectorstores import Chroma


class VectorStoreBackend(Protocol):
    """Protocol describing vector store interactions."""

    def build_from_documents(
        self,
        documents: Iterable[Any],
        embeddings: Any,
        persist_directory: Path,
        ids: List[str],
    ) -> Any:
        ...

    def load(self, persist_directory: Path, embeddings: Any) -> Any:
        ...

    def add_documents(self, store: Any, documents: Iterable[Any], ids: List[str]):
        ...

    def delete(self, store: Any, where: dict):
        ...

    def similarity_search_with_score(self, store: Any, query: str, k: int):
        ...

    def persist(self, store: Any):
        ...


class ChromaBackend:
    """Chroma implementation of the VectorStoreBackend protocol."""

    def build_from_documents(
        self,
        documents: Iterable[Any],
        embeddings: Any,
        persist_directory: Path,
        ids: List[str],
    ) -> Chroma:
        return Chroma.from_documents(
            documents=list(documents),
            embedding=embeddings,
            ids=ids,
            persist_directory=str(persist_directory),
        )

    def load(self, persist_directory: Path, embeddings: Any) -> Chroma:
        return Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
        )

    def add_documents(self, store: Chroma, documents: Iterable[Any], ids: List[str]):
        store.add_documents(list(documents), ids=ids)

    def delete(self, store: Chroma, where: dict):
        store.delete(where=where)

    def similarity_search_with_score(self, store: Chroma, query: str, k: int):
        return store.similarity_search_with_score(query, k=k)

    def persist(self, store: Chroma):
        store.persist()


DEFAULT_VECTORSTORE_BACKEND = ChromaBackend()

