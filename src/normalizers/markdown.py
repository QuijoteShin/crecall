# src/normalizers/markdown.py
"""Markdown-based content normalizer with Soft Sanitization."""

import re
import html
from typing import List, Dict, Optional

from ..contracts.normalizer import INormalizer


class MarkdownNormalizer(INormalizer):
    """
    Normalizes content to clean Markdown with dual-output strategy.

    Outputs:
    1. clean_content: Formatted Markdown for UI
    2. vectorizable_content: Sanitized text for embeddings (NO stopword removal)

    Strategy: "Soft Sanitization" (ADR approved)
    - Remove technical noise (logs, separators, mojibake)
    - Remove invisible/dangerous characters
    - Replace code blocks with [CODE_BLOCK] token
    - KEEP stopwords (neural models need grammatical structure)
    """

    # Invisible and dangerous Unicode characters
    INVISIBLE_CHARS = [
        '\u200B',  # Zero-width space
        '\u200C',  # Zero-width non-joiner
        '\u200D',  # Zero-width joiner
        '\u200E',  # Left-to-right mark
        '\u200F',  # Right-to-left mark
        '\uFEFF',  # Zero-width no-break space (BOM)
        '\u202A',  # Left-to-right embedding
        '\u202B',  # Right-to-left embedding
        '\u202C',  # Pop directional formatting
        '\u202D',  # Left-to-right override
        '\u202E',  # Right-to-left override
    ]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.use_markitdown = self.config.get('use_markitdown', True)
        self._markitdown_available = self._check_markitdown()

    def _check_markitdown(self) -> bool:
        """Check if markitdown is available."""
        if not self.use_markitdown:
            return False

        try:
            import markitdown
            return True
        except ImportError:
            return False

    def normalize(self, content: str, content_type: str = "text") -> str:
        """
        Clean and normalize content for UI display (formatted Markdown).

        Args:
            content: Raw content string
            content_type: Type hint ('text', 'html', 'code', 'markdown')

        Returns:
            Normalized Markdown string
        """
        if not content:
            return ""

        # Remove invisible characters first (security)
        content = self.remove_invisible_chars(content)

        # Handle based on content type
        if content_type == 'html':
            content = self.convert_to_markdown(content, 'html')
        elif content_type == 'code':
            # Already clean, preserve as-is
            content = self._normalize_whitespace(content, preserve_newlines=True)
        else:
            # Text or markdown
            content = self._normalize_whitespace(content, preserve_newlines=True)

        return content.strip()

    def prepare_for_vectorization(self, content: str) -> str:
        """
        Apply Soft Sanitization for embedding generation.

        ADR-compliant:
        - ✅ Remove: Visual noise, code blocks, URLs, logs
        - ❌ Keep: Stopwords, negations, prepositions, grammatical structure

        Args:
            content: Normalized markdown content

        Returns:
            Sanitized text ready for vectorization
        """
        if not content:
            return ""

        text = content

        # 1. Replace code blocks with token (preserve context)
        text = self._replace_code_blocks(text)

        # 2. Remove visual separators (===, ---, ***)
        text = re.sub(r'[-=*_]{3,}', ' ', text)

        # 3. Replace URLs with [URL] token
        text = self._replace_urls(text)

        # 4. Remove log-like patterns
        text = self._remove_log_patterns(text)

        # 5. Remove markdown formatting (but keep text)
        text = self._strip_markdown_formatting(text)

        # 6. Normalize whitespace (single spaces, max 2 newlines)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 7. Remove prompt injection attempts
        text = self._remove_prompt_injection_patterns(text)

        return text.strip()

    def remove_invisible_chars(self, text: str) -> str:
        """
        Remove zero-width spaces, RTL marks, and other invisible characters.

        Critical for security (prevents prompt injection via hidden characters).
        """
        for char in self.INVISIBLE_CHARS:
            text = text.replace(char, '')

        # Also remove control characters except newlines and tabs
        text = ''.join(
            char for char in text
            if char in '\n\r\t' or not (0 <= ord(char) < 32 or ord(char) == 127)
        )

        return text

    def convert_to_markdown(self, content: str, source_format: str) -> str:
        """
        Convert content to clean Markdown.

        Uses markitdown if available, otherwise fallback to manual conversion.

        Args:
            content: Raw content
            source_format: 'html', 'docx', 'pdf', etc.

        Returns:
            Markdown string
        """
        if source_format == 'html':
            if self._markitdown_available:
                return self._convert_with_markitdown(content, source_format)
            else:
                return self._html_to_markdown_fallback(content)
        else:
            # For other formats, try markitdown or return as-is
            if self._markitdown_available:
                return self._convert_with_markitdown(content, source_format)
            return content

    def extract_code_blocks(self, content: str) -> List[Dict]:
        """
        Extract code blocks with language tags.

        Returns:
            List of dicts: [{'language': 'python', 'code': '...', 'line': 10}]
        """
        code_blocks = []

        # Pattern for fenced code blocks: ```language\ncode\n```
        pattern = r'```(\w+)?\n(.*?)```'

        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            line_num = content[:match.start()].count('\n') + 1

            code_blocks.append({
                'language': language,
                'code': code,
                'line': line_num,
            })

        return code_blocks

    def _convert_with_markitdown(self, content: str, source_format: str) -> str:
        """Use markitdown library for conversion."""
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            # markitdown expects file paths or file-like objects
            # For string content, we'd need to adapt this
            # This is a placeholder for the actual implementation
            return content  # Fallback for now
        except Exception:
            return content

    def _html_to_markdown_fallback(self, html_content: str) -> str:
        """
        Simple HTML to Markdown conversion (fallback).
        """
        # Unescape HTML entities
        text = html.unescape(html_content)

        # Remove script and style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Convert common HTML tags to Markdown
        text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n', text, flags=re.IGNORECASE)

        text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.IGNORECASE)
        text = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', text, flags=re.IGNORECASE)
        text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text, flags=re.IGNORECASE)
        text = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', text, flags=re.IGNORECASE)

        text = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.IGNORECASE)
        text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.IGNORECASE)
        text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', text, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        return self._normalize_whitespace(text, preserve_newlines=True)

    def _replace_code_blocks(self, text: str) -> str:
        """Replace code blocks with semantic token."""
        # Fenced code blocks
        def replace_fenced(match):
            language = match.group(1) or 'code'
            return f'[CODE_BLOCK: {language}]'

        text = re.sub(r'```(\w+)?\n.*?```', replace_fenced, text, flags=re.DOTALL)

        # Inline code
        text = re.sub(r'`[^`]+`', '[CODE]', text)

        return text

    def _replace_urls(self, text: str) -> str:
        """Replace URLs with [URL] token."""
        # HTTP(S) URLs
        text = re.sub(
            r'https?://[^\s]+',
            '[URL]',
            text,
            flags=re.IGNORECASE
        )
        return text

    def _remove_log_patterns(self, text: str) -> str:
        """Remove common log patterns."""
        # [INFO], [ERROR], [DEBUG], etc.
        text = re.sub(r'\[(INFO|ERROR|DEBUG|WARN|WARNING|TRACE)\]', '', text, flags=re.IGNORECASE)

        # Timestamps: 2024-01-01 12:00:00
        text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text)

        # User joined/left messages
        text = re.sub(r'(User|Member)\s+(joined|left)\s+(channel|room)', '', text, flags=re.IGNORECASE)

        return text

    def _strip_markdown_formatting(self, text: str) -> str:
        """Remove Markdown formatting but keep text content."""
        # Headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Bold/italic
        text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)

        # Links: [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Images: ![alt](url) → alt
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)

        # Blockquotes
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

        # List markers
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)

        return text

    def _remove_prompt_injection_patterns(self, text: str) -> str:
        """Remove potential prompt injection attempts."""
        # System/Instructions keywords
        patterns = [
            r'(?i)system:\s*ignore\s+previous',
            r'(?i)ignore\s+all\s+previous\s+instructions',
            r'(?i)disregard\s+all\s+previous',
            r'(?i)forget\s+everything',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text)

        return text

    def _normalize_whitespace(self, text: str, preserve_newlines: bool = True) -> str:
        """Normalize whitespace while optionally preserving paragraph breaks."""
        if preserve_newlines:
            # Collapse multiple newlines to max 2 (paragraph break)
            text = re.sub(r'\n{3,}', '\n\n', text)

            # Remove trailing whitespace from each line
            lines = text.split('\n')
            lines = [line.rstrip() for line in lines]
            text = '\n'.join(lines)
        else:
            # Collapse all whitespace to single spaces
            text = ' '.join(text.split())

        return text
