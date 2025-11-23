"""Parallel embedding helpers and caching layer."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Sequence


logger = logging.getLogger("homerag")


class LRUCache:
    """Simple LRU cache for embedding results."""

    def __init__(self, max_size: int = 256):
        self.max_size = max(1, max_size)
        self._data: OrderedDict[str, List[float]] = OrderedDict()

    def get(self, key: str) -> Optional[List[float]]:
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key: str, value: List[float]):
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
            return
        self._data[key] = value
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)


@dataclass
class EmbeddingPipelineConfig:
    batch_size: int
    max_concurrency: int
    cache_size: int = 0


class ParallelEmbeddingEngine:
    """Execute embedding requests with controlled concurrency and batching."""

    def __init__(self, base_embeddings, config: EmbeddingPipelineConfig):
        self._base = base_embeddings
        self._config = config
        self._cache = LRUCache(config.cache_size) if config.cache_size else None
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing: list[tuple[int, str]] = []

        for idx, text in enumerate(texts):
            cached = self._cache.get(text) if self._cache else None
            if cached is not None:
                results[idx] = cached
            else:
                missing.append((idx, text))

        if missing:
            batches = [
                missing[i : i + self._config.batch_size]
                for i in range(0, len(missing), self._config.batch_size)
            ]

            async def process_batch(batch: list[tuple[int, str]]):
                async with self._semaphore:
                    batch_texts = [text for _idx, text in batch]
                    vectors = await asyncio.to_thread(
                        self._base.embed_documents, batch_texts
                    )
                    for (idx, text), vector in zip(batch, vectors):
                        results[idx] = vector
                        if self._cache is not None:
                            self._cache.put(text, vector)

            await asyncio.gather(*(process_batch(batch) for batch in batches))

        return [vector or [] for vector in results]

    async def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        cached = self._cache.get(text) if self._cache else None
        if cached is not None:
            return cached

        async with self._semaphore:
            vector = await asyncio.to_thread(self._base.embed_query, text)
        if self._cache is not None:
            self._cache.put(text, vector)
        return vector

    def embed_documents_sync(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.embed_documents(texts))

        # Fallback for synchronous calls invoked inside an existing event loop.
        # We bypass the async pipeline to avoid nested event loop execution while
        # still reusing the base embedding implementation and cache.
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing: list[tuple[int, str]] = []

        for idx, text in enumerate(texts):
            cached = self._cache.get(text) if self._cache else None
            if cached is not None:
                results[idx] = cached
            else:
                missing.append((idx, text))

        if missing:
            vectors = self._base.embed_documents([text for _idx, text in missing])
            for (idx, text), vector in zip(missing, vectors):
                results[idx] = vector
                if self._cache is not None:
                    self._cache.put(text, vector)

        return [vector or [] for vector in results]

    def embed_query_sync(self, text: str) -> List[float]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.embed_query(text))

        if not text:
            return []

        cached = self._cache.get(text) if self._cache else None
        if cached is not None:
            return cached

        vector = self._base.embed_query(text)
        if self._cache is not None:
            self._cache.put(text, vector)
        return vector

