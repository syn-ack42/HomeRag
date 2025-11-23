"""Indexing state manager for incremental embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

from app_core.bot_registry import bot_paths


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_file_hashes(bot_id: str) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    data_dir = bot_paths(bot_id)["data"]
    if not data_dir.exists():
        return hashes
    for file in data_dir.rglob("*"):
        if file.is_file():
            rel = file.relative_to(data_dir).as_posix()
            hashes[rel] = _hash_file(file)
    return hashes


@dataclass
class IndexState:
    chunk_size: int
    chunk_overlap: int
    file_hashes: Dict[str, str] = field(default_factory=dict)


class IndexingManager:
    """Load and persist index metadata for delta rebuilds."""

    def __init__(self, embedding_dir: Path):
        self.embedding_dir = embedding_dir
        self.meta_file = embedding_dir / "index_state.json"

    def load(self) -> IndexState:
        if not self.meta_file.exists():
            return IndexState(chunk_size=-1, chunk_overlap=-1, file_hashes={})
        try:
            data = json.loads(self.meta_file.read_text())
        except Exception:
            return IndexState(chunk_size=-1, chunk_overlap=-1, file_hashes={})
        return IndexState(
            chunk_size=int(data.get("chunk_size", -1)),
            chunk_overlap=int(data.get("chunk_overlap", -1)),
            file_hashes=data.get("file_hashes", {}) or {},
        )

    def save(self, state: IndexState):
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_size": state.chunk_size,
            "chunk_overlap": state.chunk_overlap,
            "file_hashes": state.file_hashes,
        }
        self.meta_file.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def diff_hashes(new_hashes: Dict[str, str], old_hashes: Dict[str, str]) -> Tuple[set, set]:
        added_or_changed = {path for path, digest in new_hashes.items() if old_hashes.get(path) != digest}
        removed = set(old_hashes.keys()) - set(new_hashes.keys())
        return added_or_changed, removed

