# src/importers/chatgpt.py
"""ChatGPT export importer."""

import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Iterator, Optional, Dict, Any, List

from ..contracts.importer import IImporter
from ..models.normalized import NormalizedMessage, Attachment


class ChatGPTImporter(IImporter):
    """
    Imports OpenAI ChatGPT export format.

    Expected structure:
    - conversations.json (main conversation data)
    - user.json (user metadata)
    - file_*.png/jpg (attachments)
    """

    SUPPORTED_FORMATS = ['chatgpt', 'openai']

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def supports_format(self, format_id: str) -> bool:
        """Check if this importer handles the format."""
        return format_id.lower() in self.SUPPORTED_FORMATS

    def get_metadata(self, source: Path) -> dict:
        """Extract metadata without full import."""
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        if not zipfile.is_zipfile(source):
            raise ValueError(f"Source is not a ZIP file: {source}")

        metadata = {
            'format': 'chatgpt',
            'total_conversations': 0,
            'date_range': None,
            'user_info': None,
        }

        with zipfile.ZipFile(source, 'r') as zf:
            # Read user info
            if 'user.json' in zf.namelist():
                user_data = json.loads(zf.read('user.json'))
                metadata['user_info'] = user_data

            # Count conversations
            if 'conversations.json' in zf.namelist():
                conversations = json.loads(zf.read('conversations.json'))
                metadata['total_conversations'] = len(conversations)

                # Get date range
                if conversations:
                    timestamps = [
                        c.get('create_time', 0)
                        for c in conversations
                        if c.get('create_time')
                    ]
                    if timestamps:
                        metadata['date_range'] = {
                            'start': datetime.fromtimestamp(min(timestamps)).isoformat(),
                            'end': datetime.fromtimestamp(max(timestamps)).isoformat(),
                        }

        return metadata

    def import_data(self, source: Path) -> Iterator[NormalizedMessage]:
        """Import and yield normalized messages."""
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        if not zipfile.is_zipfile(source):
            raise ValueError(f"Source is not a ZIP file: {source}")

        with zipfile.ZipFile(source, 'r') as zf:
            # Load conversations
            if 'conversations.json' not in zf.namelist():
                raise ValueError("conversations.json not found in ZIP")

            conversations = json.loads(zf.read('conversations.json'))

            # Process each conversation
            for conv in conversations:
                yield from self._process_conversation(conv, zf)

    def _process_conversation(
        self,
        conversation: dict,
        zipfile_ref: zipfile.ZipFile
    ) -> Iterator[NormalizedMessage]:
        """
        Process a single conversation and yield messages.

        ChatGPT stores conversations as a tree (mapping) of messages.
        We need to walk the tree to extract the linear conversation flow.
        """
        conv_id = conversation.get('id', conversation.get('conversation_id', 'unknown'))
        conv_title = conversation.get('title', 'Untitled')
        mapping = conversation.get('mapping', {})

        # Walk the message tree
        for node_id, node_data in mapping.items():
            message_data = node_data.get('message')

            if not message_data:
                continue

            # Skip system messages with no content
            if message_data.get('metadata', {}).get('is_visually_hidden_from_conversation'):
                continue

            author = message_data.get('author', {})
            role = author.get('role', 'unknown')

            content_data = message_data.get('content', {})
            content_type = content_data.get('content_type', 'text')

            # Extract text content
            parts = content_data.get('parts', [])
            if not parts:
                content_text = ""
            elif isinstance(parts[0], str):
                content_text = parts[0]
            else:
                content_text = str(parts[0])

            # Skip empty messages
            if not content_text.strip():
                continue

            # Parse timestamp
            create_time = message_data.get('create_time')
            if create_time:
                timestamp = datetime.fromtimestamp(create_time)
            else:
                timestamp = datetime.now()

            # Handle attachments (if referenced in message)
            attachments = self._extract_attachments(message_data, zipfile_ref)

            # Build normalized message
            normalized = NormalizedMessage(
                id=message_data['id'],
                conversation_id=conv_id,
                author_role=self._normalize_role(role),
                content=content_text,
                content_type=content_type,
                timestamp=timestamp,
                metadata={
                    'conversation_title': conv_title,
                    'original_role': role,
                    'author_name': author.get('name'),
                    'status': message_data.get('status'),
                    'weight': message_data.get('weight', 1.0),
                },
                attachments=attachments,
                parent_message_id=node_data.get('parent'),
            )

            yield normalized

    def _normalize_role(self, role: str) -> str:
        """
        Normalize ChatGPT roles to universal format.

        ChatGPT roles: 'system', 'user', 'assistant', 'tool'
        """
        role_lower = role.lower()
        if role_lower in ['user', 'assistant', 'system', 'tool']:
            return role_lower
        return 'system'  # Fallback

    def _extract_attachments(
        self,
        message_data: dict,
        zipfile_ref: zipfile.ZipFile
    ) -> List[Attachment]:
        """
        Extract attachment references from message.

        ChatGPT stores file references in message metadata.
        """
        attachments = []

        # Check for file attachments in metadata
        metadata = message_data.get('metadata', {})
        attachments_meta = metadata.get('attachments', [])

        for att_meta in attachments_meta:
            file_id = att_meta.get('id', att_meta.get('file_id'))
            filename = att_meta.get('name', f"file_{file_id}")
            content_type = att_meta.get('mime_type', 'application/octet-stream')

            # Look for matching file in ZIP
            matching_files = [
                f for f in zipfile_ref.namelist()
                if file_id in f
            ]

            local_path = matching_files[0] if matching_files else None

            attachment = Attachment(
                id=file_id,
                filename=filename,
                content_type=content_type,
                local_path=local_path,
            )
            attachments.append(attachment)

        return attachments
