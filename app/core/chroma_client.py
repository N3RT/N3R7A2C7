from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from app.core.config_loader import load_system_config, SystemConfigError


_chroma_client: Optional[ClientAPI] = None


def get_chroma_client() -> ClientAPI:
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    cfg = load_system_config()
    paths = cfg.get("paths", {}) or {}
    chroma_dir = paths.get("chromadb_dir", "./data/chromadb")
    chroma_path = Path(chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.Client(
        Settings(
            is_persistent=True,
            persist_directory=str(chroma_path),
        )
    )
    return _chroma_client
