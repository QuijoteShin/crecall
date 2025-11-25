# Chat Recall

**Deep Research Chat Recall** - Sistema cognitivo para indexar y buscar historial de conversaciones usando embeddings semánticos.

```
 ██████╗██████╗ ███████╗ ██████╗
██╔════╝██╔══██╗██╔════╝██╔════╝
██║     ██████╔╝█████╗  ██║
██║     ██╔══██╗██╔══╝  ██║
╚██████╗██║  ██║███████╗╚██████╗
 ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝  v0.2.0
```

## Highlights

| Feature | Description |
|---------|-------------|
| **Vector First** | Busqueda por significado, no por tags |
| **Nomic Embed v1.5** | MRL 384 dims, query/document prefixes |
| **TUI Interactivo** | Navegacion completa, historial, export |
| **Checkpoint Rebuild** | Resume donde quedo si se interrumpe |
| **Low VRAM** | <500MB GPU, carga serial |
| **Pluggable** | Swap components via YAML |

## Installation

```bash
# Clonar y entrar
cd /var/www/drecall

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -e .

# Verificar
python crec.py version
```

### Requirements

- Python 3.10+
- CUDA (opcional, recomendado)
- ~2GB disk para modelos
- ~500MB VRAM (o CPU mode)

## Quick Start

```bash
# 1. Indexar historial de ChatGPT
python crec.py init process/in/chatgpt_history.zip

# 2. Lanzar TUI interactivo
python crec.py interactive
```

## Search Syntax

### Basic Search
```
query           → Busqueda semantica pura
"exact phrase"  → Busqueda de frase exacta (boost)
```

### Topic Search (Vector First)
```
..sql..              → Buscar TODO sobre SQL/databases
..python..           → Todo sobre Python
bintel ..sql..       → "bintel" en contexto de SQL
auth ..security..    → "auth" en contexto de seguridad
```

### How Vector First Works

```
ANTES (tags literales):
  "bintel ..sql.." → filter by topic="sql" → 0 results
                     (topics were ['bigint', 'code'])

AHORA (semantico):
  "bintel ..sql.." → search "bintel sql" by MEANING → finds it!
                     (Nomic entiende que bigint = SQL)
```

## CLI Commands

```bash
# Indexar datos
python crec.py init data.zip [--profile lite|express] [--cpu-only]

# TUI interactivo (recomendado)
python crec.py interactive

# Busqueda CLI
python crec.py search "query" [--limit 20] [--mode vector|text]

# Rebuild vectores (con checkpoint)
python crec.py rebuild --vectors-only --yes [--no-resume]

# Re-indexar (cambiar modelo)
python crec.py reindex [--force] [--limit 1000]

# Estadisticas
python crec.py stats

# Version
python crec.py version
```

## TUI Features

### Navigation

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Cambiar panel |
| `↑↓` (en input) | Historial de busquedas (como bash) |
| `↑↓` (en resultados) | Navegar lista |
| `Enter` | Ver conversacion completa |
| `1-5` | Seleccion rapida de resultado |
| `ESC` | Volver atras |

### Actions

| Key | Action |
|-----|--------|
| `Ctrl+F` | Buscar DENTRO de conversacion |
| `n` / `N` | Siguiente/anterior match |
| `Ctrl+J` | Saltar al mensaje matcheado |
| `Ctrl+M` | Cargar mas resultados (infinite scroll) |
| `Ctrl+P` | Exportar conversacion a Markdown |
| `Ctrl+H` | Mostrar ayuda |
| `Ctrl+O` | Opciones de configuracion |
| `Ctrl+C` | Salir |

### Slash Commands

```
/help              Ver ayuda completa
/stats             Estadisticas de la base de datos
/theme dark|light  Cambiar tema y guardar
/limit 20          Cambiar limite de resultados
/lang es|en        Cambiar idioma
/files             Listar archivos indexados (WIP)
```

### Query History

El input guarda historial como bash:
- `↑` - Query anterior
- `↓` - Query siguiente
- Se persiste en `.drecall/search_history.txt`

## Configuration

### User Config (~/.drecall/config.toml)

```toml
[personality]
name = "CognitiveAssistant"
language = "es"

[prompts]
welcome = "Chat Recall v{version} | {total_messages} mensajes"
search_intro = "Encontre estas conversaciones:"
no_results = "No encontre nada. Reformula?"
loading = "Buscando..."

[search]
default_limit = 10
min_similarity = 0.3

[ui]
theme = "dark"  # dark, light, auto
result_preview_lines = 3
```

### System Config (config/default.yaml)

```yaml
components:
  vectorizer:
    class: "src.vectorizers.nomic_embed.NomicEmbedVectorizer"
    config:
      model: "nomic-ai/nomic-embed-text-v1.5"
      dimension: 384      # MRL truncation (768 full)
      batch_size: 32
      device: "cuda"

  classifier:
    class: "src.classifiers.transformer.TransformerClassifier"
    config:
      model: "valhalla/distilbart-mnli-12-1"
      batch_size: 32
      device: "cuda"

  storage:
    class: "src.storage.sqlite_vector.SQLiteVectorStorage"
    config:
      database: "data/recall.db"
      dimension: 384

memory:
  concurrent_models: false
  force_gc_between_stages: true
  cuda_empty_cache_between_stages: true
```

## Profiles

| Profile | Classifier | Vectorizer | Time | Use Case |
|---------|------------|------------|------|----------|
| `default` | DistilBART | Nomic v1.5 | ~15min/10k | Full AI |
| `lite` | Rule-based | Nomic v1.5 | ~5min/10k | Fast |
| `express` | Skip | Nomic v1.5 | ~2min/10k | Ultra-fast |

```bash
python crec.py init data.zip --profile express
```

## Rebuild & Checkpoint

Rebuild soporta checkpoint automatico:

```bash
# Ejecutar rebuild
python crec.py rebuild --vectors-only --yes

# Si se interrumpe (Ctrl+C), ejecutar de nuevo:
python crec.py rebuild --vectors-only --yes
# → Detecta vectores ya convertidos y continua

# Forzar re-vectorizar todo:
python crec.py rebuild --vectors-only --yes --no-resume
```

El checkpoint funciona detectando la dimension del vector:
- Si vector tiene 384 dims (Nomic) → skip
- Si vector tiene otra dimension o NULL → re-vectorizar

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CREC PIPELINE                              │
└──────────────────────────────────────────────────────────────┘

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Import  │──▶│Normalize │──▶│ Classify │──▶│Vectorize │
│(ChatGPT) │   │(Markdown)│   │(DistilBART)  │(Nomic)   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                   │              │
                              GPU Stage 1    GPU Stage 2
                                   │              │
                                   ▼              ▼
                    ┌─────────────────────────────────────┐
                    │      SQLite + sqlite-vec            │
                    │   FTS5 (keywords) + KNN (vectors)   │
                    └─────────────────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         SEARCH ENGINE               │
                    │   Vector First + Keyword Bonus      │
                    └─────────────────────────────────────┘
```

### Search Score Calculation

```python
final_score = vector_similarity + keyword_bonus + topic_bonus

# vector_similarity: 0.0 - 1.0 (cosine)
# keyword_bonus:     0.0 - 0.3 (exact matches)
# topic_bonus:       0.0 - 0.15 (tag matches)
```

## Memory Usage

| Component | VRAM | RAM | Notes |
|-----------|------|-----|-------|
| DistilBART | ~400 MB | ~200 MB | Classifier |
| Nomic Embed | ~500 MB | ~300 MB | Vectorizer |
| **Peak** | ~500 MB | ~500 MB | Serial loading |

CPU mode available with `--cpu-only`.

## Supported Formats

| Format | Status | Importer |
|--------|--------|----------|
| ChatGPT Export (.zip) | ✅ | `ChatGPTImporter` |
| Claude Export | 🔜 | Planned |
| WhatsApp Export | 🔜 | Planned |
| PDF Documents | 🔜 | Planned |
| Generic JSON | 🔜 | Planned |

## Project Structure

```
crec/
├── crec.py                 # Entry point
├── src/
│   ├── __init__.py         # Version, app name
│   ├── contracts/          # Abstract interfaces (IVectorizer, etc.)
│   ├── models/             # Data models (NormalizedMessage, etc.)
│   ├── core/               # Pipeline, Registry, Config
│   ├── importers/          # Format-specific importers
│   ├── normalizers/        # Content normalizers
│   ├── classifiers/        # Intent classifiers
│   ├── vectorizers/        # Embedding generators
│   │   ├── nomic_embed.py      # Nomic v1.5 (default)
│   │   ├── sentence_transformer.py
│   │   └── embedding_gemma.py  # Google 300M
│   ├── storage/            # SQLite + sqlite-vec
│   ├── tui/                # Textual TUI
│   └── cli/                # Typer CLI
├── config/
│   ├── default.yaml        # Full AI pipeline
│   ├── lite.yaml           # Rule-based classifier
│   └── express.yaml        # Vectors only
└── data/
    └── recall.db           # SQLite database
```

## Extending

### Add Vectorizer

```python
# src/vectorizers/my_vectorizer.py
from src.contracts.vectorizer import IVectorizer
import numpy as np

class MyVectorizer(IVectorizer):
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None

    def load(self):
        # Load your model
        pass

    def unload(self):
        # Free resources
        pass

    def vectorize(self, text: str) -> np.ndarray:
        # Return embedding
        pass

    def vectorize_batch(self, texts: list) -> np.ndarray:
        # Batch processing
        pass

    def vectorize_query(self, query: str) -> np.ndarray:
        # Optional: query-specific embedding
        # Nomic uses "search_query:" prefix
        return self.vectorize(query)

    def get_embedding_dim(self) -> int:
        return 384
```

### Add Importer

```python
# src/importers/my_importer.py
from src.contracts.importer import IImporter
from src.models.normalized import NormalizedMessage

class MyImporter(IImporter):
    def supports_format(self, format_id: str) -> bool:
        return format_id == "my_format"

    def import_data(self, source: Path) -> Iterator[NormalizedMessage]:
        # Yield normalized messages
        yield NormalizedMessage(
            id="unique-id",
            conversation_id="conv-id",
            author_role="user",  # or "assistant"
            content="Message content",
            content_type="text",
            timestamp=datetime.now(),
            metadata={"source": "my_format"}
        )
```

## Troubleshooting

### Out of VRAM
```bash
# Use CPU mode
python crec.py init data.zip --cpu-only

# Or use smaller batch size
python crec.py init data.zip --batch-size 8
```

### Search returns 0 results
```bash
# Check if vectors exist
python crec.py stats

# Rebuild vectors if needed
python crec.py rebuild --vectors-only --yes
```

### TUI MarkupError
Fixed in v0.2.0 - content is now escaped using `Text.append()`.

## Roadmap

- [x] Vector First search strategy
- [x] Nomic Embed v1.5 (MRL 384d)
- [x] Checkpoint/resume rebuild
- [x] TUI with Rich Text (escaped)
- [x] Query history (bash-like)
- [x] Find in conversation (Ctrl+F)
- [x] Export to Markdown (Ctrl+P)
- [ ] Zero-shot topic classification (Nomic)
- [ ] HCS graph visualization
- [ ] Claude importer
- [ ] WhatsApp importer
- [ ] Web UI (BintelX integration)

## License

MIT

---

**Chat Recall** - *Find what you talked about, not what you tagged.*
