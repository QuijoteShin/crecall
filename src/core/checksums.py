# src/core/checksums.py
"""Checksum utilities for deduplication and file tracking."""

import hashlib
from pathlib import Path


def compute_content_hash(content: str) -> str:
    """
    Compute SHA256 hash of content for deduplication.

    Args:
        content: Text content to hash

    Returns:
        Hex string of SHA256 hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of entire file.

    Args:
        file_path: Path to file

    Returns:
        Hex string of SHA256 hash
    """
    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_chunk_hash(text: str, chunk_size: int = 512) -> list[str]:
    """
    Compute hashes of text chunks for partial matching.

    Useful for detecting similar content even if not exact match.

    Args:
        text: Text to chunk and hash
        chunk_size: Characters per chunk

    Returns:
        List of hash strings
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
        chunks.append(chunk_hash)

    return chunks
