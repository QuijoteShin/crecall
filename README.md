# Chat Recall

**Deep Research Chat Recall** - Sistema cognitivo para indexar y buscar historial de conversaciones usando embeddings semánticos.

```
 ██████╗██████╗ ███████╗ ██████╗
██╔════╝██╔══██╗██╔════╝██╔════╝
██║     ██████╔╝█████╗  ██║
██║     ██╔══██╗██╔══╝  ██║
╚██████╗██║  ██║███████╗╚██████╗
 ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝
```

## Features

- **Vector Truth Strategy** - Búsqueda por significado semántico, no por tags literales
- **Nomic Embed v1.5** - Embeddings con Matryoshka Representation Learning (384 dims)
- **Low VRAM Pipeline** - Operación serial para GPUs <8GB
- **Pluggable Architecture** - Componentes intercambiables via YAML
- **TUI Interactivo** - Interfaz de terminal con navegación completa

## Quick Start

```bash
# Activar entorno
source .venv/bin/activate

# Indexar historial de ChatGPT
python crec.py init process/in/chatgpt_history.zip

# Modo interactivo (recomendado)
python crec.py interactive
```

## Search Syntax

| Syntax | Descripcion | Ejemplo |
|--------|-------------|---------|
| `query` | Busqueda semantica | `authentication jwt` |
| `..topic..` | Buscar por significado del topic | `..sql..` |
| `query ..topic..` | Mixta: query + contexto | `bintel ..sql..` |

### Vector First Strategy

La busqueda usa el **vector semantico como fuente de verdad**:

```
"bintel ..sql.." → Busca "bintel sql" por SIGNIFICADO
                   No filtra por tags literales
                   Encuentra aunque topics sean ['bigint', 'code']
```

## CLI Commands

```bash
# Indexar datos
python crec.py init data.zip [--profile lite|express]

# Busqueda CLI
python crec.py search "query" [--limit 20]

# Rebuild vectores (con checkpoint)
python crec.py rebuild --vectors-only --yes

# Estadisticas
python crec.py stats

# Version
python crec.py version
```

## TUI Navigation

| Key | Action |
|-----|--------|
| `Tab` | Cambiar panel |
| `↑↓` | Navegar resultados / historial |
| `Enter` | Ver conversacion |
| `1-5` | Seleccion rapida |
| `ESC` | Volver |
| `Ctrl+F` | Buscar en conversacion |
| `Ctrl+M` | Cargar mas resultados |
| `Ctrl+P` | Exportar a Markdown |

## Configuration

### User Config (~/.drecall/config.toml)

```toml
[personality]
language = "es"

[search]
default_limit = 10
min_similarity = 0.3

[ui]
theme = "dark"
```

### System Config (config/default.yaml)

```yaml
components:
  vectorizer:
    class: "src.vectorizers.nomic_embed.NomicEmbedVectorizer"
    config:
      model: "nomic-ai/nomic-embed-text-v1.5"
      dimension: 384
      device: "cuda"

  classifier:
    class: "src.classifiers.transformer.TransformerClassifier"
    config:
      model: "valhalla/distilbart-mnli-12-1"
      device: "cuda"

  storage:
    class: "src.storage.sqlite_vector.SQLiteVectorStorage"
    config:
      database: "data/recall.db"
      dimension: 384
```

## Profiles

| Profile | Classifier | Vectorizer | Use Case |
|---------|------------|------------|----------|
| `default` | DistilBART | Nomic v1.5 | Full AI pipeline |
| `lite` | Rule-based | Nomic v1.5 | Fast, less GPU |
| `express` | Skip | Nomic v1.5 | Ultra-fast indexing |

```bash
python crec.py init data.zip --profile lite
```

## Architecture

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Import  │──▶│Normalize │──▶│ Classify │──▶│Vectorize │
│(ChatGPT) │   │(Markdown)│   │(DistilBART)  │(Nomic)   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                   │              │
                              GPU Stage 1    GPU Stage 2
                                   │              │
                                   ▼              ▼
                              ┌─────────────────────────┐
                              │  SQLite + sqlite-vec    │
                              │  (FTS5 + Vector KNN)    │
                              └─────────────────────────┘
```

## Memory Usage

| Component | VRAM | RAM |
|-----------|------|-----|
| DistilBART | ~400 MB | ~200 MB |
| Nomic Embed | ~500 MB | ~300 MB |
| **Peak** | ~500 MB | ~500 MB |

Serial loading ensures low VRAM usage.

## Project Structure

```
crec/
├── src/
│   ├── contracts/      # Abstract interfaces
│   ├── vectorizers/    # Nomic, SentenceTransformer, Gemma
│   ├── classifiers/    # DistilBART, Rule-based
│   ├── storage/        # SQLite + sqlite-vec
│   ├── tui/            # Textual TUI
│   └── cli/            # Typer CLI
├── config/             # YAML profiles
└── data/               # Database
```

## Extending

### Add Vectorizer

```python
# src/vectorizers/my_vectorizer.py
from src.contracts.vectorizer import IVectorizer

class MyVectorizer(IVectorizer):
    def vectorize(self, text: str) -> np.ndarray:
        ...
    def vectorize_query(self, query: str) -> np.ndarray:
        # Optional: query-specific embedding
        return self.vectorize(query)
```

### Add Importer

```python
# src/importers/my_importer.py
from src.contracts.importer import IImporter

class MyImporter(IImporter):
    def supports_format(self, format_id: str) -> bool:
        return format_id == "my_format"

    def import_data(self, source: Path) -> Iterator[NormalizedMessage]:
        ...
```

## Roadmap

- [x] Vector First search strategy
- [x] Nomic Embed v1.5 (MRL 384d)
- [x] Checkpoint/resume rebuild
- [x] TUI with Rich Text
- [ ] Zero-shot topic classification
- [ ] HCS graph visualization
- [ ] Claude importer
- [ ] WhatsApp importer
- [ ] Web UI

## License

MIT
