# src/core/user_config.py
"""User configuration loader (.toml format)."""

import toml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PersonalityConfig:
    """User personality settings."""
    name: str = "CognitiveAssistant"
    style: str = "technical"
    language: str = "es"
    response_length: str = "concise"


@dataclass
class PromptsConfig:
    """User-defined prompts."""
    search_intro: str = "Encontré estas conversaciones relacionadas:"
    no_results: str = "No encontré nada relevante. ¿Reformulas?"
    loading: str = "Buscando en tu memoria cognitiva..."
    welcome: str = "DRecall v{version} | {total_messages} mensajes indexados"


@dataclass
class MemoryConfig:
    """Memory management settings."""
    context_window: int = 5
    max_scroll_cache: int = 100
    auto_expand: bool = True


@dataclass
class SearchConfig:
    """Search settings."""
    default_limit: int = 5
    min_similarity: float = 0.3
    auto_highlight: bool = True
    show_scores: bool = True


@dataclass
class UIConfig:
    """UI settings."""
    theme: str = "dark"
    result_preview_lines: int = 3
    show_metadata: bool = True
    keyboard_shortcuts: bool = True


@dataclass
class AIConnectionsConfig:
    """External AI service connections."""
    gemini_api_key: Optional[str] = None
    chatgpt_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None


@dataclass
class AttachmentsConfig:
    """Attachment handling settings."""
    index_images: bool = True
    index_pdfs: bool = True
    ocr_enabled: bool = False
    max_file_size_mb: int = 50


@dataclass
class UserConfig:
    """Complete user configuration."""
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    ai_connections: AIConnectionsConfig = field(default_factory=AIConnectionsConfig)
    attachments: AttachmentsConfig = field(default_factory=AttachmentsConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'UserConfig':
        """
        Load user configuration from .toml file.

        Creates default config if it doesn't exist.

        Args:
            config_path: Path to config file. If None, uses .drecall/config.toml

        Returns:
            UserConfig instance
        """
        if config_path is None:
            # ALWAYS use local config first (project-specific)
            config_path = Path(".drecall/config.toml")

            # Fallback to home directory
            if not config_path.exists():
                home_config = Path.home() / ".drecall" / "config.toml"
                if home_config.exists():
                    config_path = home_config

        # If still doesn't exist, create default config
        if not config_path.exists():
            config = cls()
            config.save(config_path)  # Save defaults
            return config

        # Load TOML
        try:
            data = toml.load(config_path)
        except Exception as e:
            # If corrupt, return defaults
            print(f"Warning: Could not load config ({e}), using defaults")
            return cls()

        # Parse sections
        return cls(
            personality=PersonalityConfig(**data.get('personality', {})),
            prompts=PromptsConfig(**data.get('prompts', {})),
            memory=MemoryConfig(**data.get('memory', {})),
            search=SearchConfig(**data.get('search', {})),
            ui=UIConfig(**data.get('ui', {})),
            ai_connections=AIConnectionsConfig(**data.get('ai_connections', {})),
            attachments=AttachmentsConfig(**data.get('attachments', {})),
        )

    def save(self, config_path: Optional[Path] = None):
        """Save configuration to file."""
        if config_path is None:
            # ALWAYS save to local config (same priority as load)
            config_path = Path(".drecall/config.toml")

        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'personality': vars(self.personality),
            'prompts': vars(self.prompts),
            'memory': vars(self.memory),
            'search': vars(self.search),
            'ui': vars(self.ui),
            'ai_connections': vars(self.ai_connections),
            'attachments': vars(self.attachments),
        }

        with open(config_path, 'w') as f:
            toml.dump(data, f)
