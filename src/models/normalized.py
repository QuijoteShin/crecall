# src/models/normalized.py
"""Universal message format (format-agnostic)."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class Attachment:
    """File attachment reference."""

    id: str
    filename: str
    content_type: str  # 'image/png', 'application/pdf', etc.
    size_bytes: Optional[int] = None
    url: Optional[str] = None  # External URL if applicable
    local_path: Optional[str] = None  # Path in extracted archive


@dataclass
class NormalizedMessage:
    """
    Universal message format that ALL importers must produce.

    This is the contract between importers and the rest of the system.
    """

    id: str
    conversation_id: str
    author_role: str  # 'user', 'assistant', 'system'
    content: str  # Main text content
    content_type: str  # 'text', 'code', 'image_ref', 'tool_call'
    timestamp: datetime

    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Attachment] = field(default_factory=list)

    parent_message_id: Optional[str] = None
    content_hash: Optional[str] = None  # SHA256 of content for deduplication

    def __post_init__(self):
        """Validate fields."""
        if self.author_role not in ['user', 'assistant', 'system', 'tool']:
            raise ValueError(f"Invalid author_role: {self.author_role}")

        if not self.id or not self.conversation_id:
            raise ValueError("id and conversation_id are required")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'author_role': self.author_role,
            'content': self.content,
            'content_type': self.content_type,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'attachments': [
                {
                    'id': a.id,
                    'filename': a.filename,
                    'content_type': a.content_type,
                    'size_bytes': a.size_bytes,
                    'url': a.url,
                    'local_path': a.local_path,
                }
                for a in self.attachments
            ],
            'parent_message_id': self.parent_message_id,
            'content_hash': self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NormalizedMessage':
        """Create from dictionary."""
        attachments = [
            Attachment(
                id=a['id'],
                filename=a['filename'],
                content_type=a['content_type'],
                size_bytes=a.get('size_bytes'),
                url=a.get('url'),
                local_path=a.get('local_path'),
            )
            for a in data.get('attachments', [])
        ]

        return cls(
            id=data['id'],
            conversation_id=data['conversation_id'],
            author_role=data['author_role'],
            content=data['content'],
            content_type=data['content_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
            attachments=attachments,
            parent_message_id=data.get('parent_message_id'),
            content_hash=data.get('content_hash'),
        )
