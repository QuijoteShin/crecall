# DRecall - Guía de Inicio Rápido

## Instalación (Solo Primera Vez)

```bash
cd /var/www/drecall

# Activar entorno virtual
source .venv/bin/activate

# Verificar instalación
python drecall.py version
```

## 1. Indexar tu Historial de ChatGPT

```bash
# Profile recomendado: lite (rápido, rule-based classifier)
python drecall.py init process/in/chatgtp_history.zip --profile lite

# Resultado:
# ✓ 19,078 mensajes indexados en ~50 segundos
# Base de datos: data/recall.db (150 MB)
```

## 2. Modo Interactivo (TUI) ⭐ RECOMENDADO

```bash
python drecall.py interactive
```

### Búsquedas:

**Búsqueda Híbrida (semántica + keywords):**
```
> arquitectura de software
```

**Búsqueda por Tópicos:**
```
> ..python..
> ..energía..
> ..blockchain..
```

**Slash Commands:**
```
> /stats          # Estadísticas
> /help           # Ayuda
> /theme light    # Cambiar a tema claro
> /limit 10       # Cambiar límite de resultados
```

### Navegación:

| Tecla | Acción |
|-------|--------|
| `Tab` | Cambiar entre input y resultados |
| `↑↓` | Navegar lista |
| `Enter` | Ver conversación completa (auto-scroll al match) |
| `1-5` | Selección rápida de resultado |
| `ESC` | Volver atrás |

### Acciones:

| Tecla | Acción |
|-------|--------|
| `Ctrl+H` | Mostrar ayuda/palette |
| `Ctrl+J` | Saltar al mensaje matcheado |
| `Ctrl+M` | Cargar más resultados (infinite scroll) |
| `Ctrl+P` | Exportar conversación a Markdown |
| `Ctrl+O` | Ver opciones de configuración |
| `Ctrl+C` | Salir |

## 3. Búsquedas desde CLI (Sin TUI)

```bash
# Búsqueda híbrida
python drecall.py search "optimización de bases de datos"

# Búsqueda por tópicos
python drecall.py search "..python.."

# Búsqueda por texto (keywords)
python drecall.py search "react hooks" --mode text

# Limitar resultados
python drecall.py search "machine learning" --limit 20
```

## 4. Estadísticas

```bash
python drecall.py stats
```

## 5. Re-indexar (Cambiar Modelo Sin Reimportar)

```bash
# Cambiar vectorizer en config/default.yaml
vim config/default.yaml

# Re-vectorizar todo
python drecall.py reindex

# Probar con subset primero
python drecall.py reindex --limit 1000
```

## Configuración Personalizada

### Archivo: `.drecall/config.toml`

Se crea automáticamente al lanzar el TUI. Puedes editarlo manualmente:

```toml
[ui]
theme = "dark"  # dark, light, auto
result_preview_lines = 3

[search]
default_limit = 5
min_similarity = 0.3

[personality]
name = "CognitiveAssistant"
language = "es"
```

### Cambiar desde el TUI:

```bash
# Lanza TUI
python drecall.py interactive

# Escribe comandos:
/theme light      # Cambiar tema
/limit 10         # Cambiar límite
/lang en          # Cambiar idioma

# Los cambios se guardan automáticamente
```

## Profiles Disponibles

### `lite` ⭐ RECOMENDADO
- Rule-based classifier (sin GPU)
- Sentence Transformers vectorizer
- **Carga en ~1 minuto**

```bash
python drecall.py init data.zip --profile lite
```

### `default`
- Transformer classifier (distilbart-mnli)
- Mejor calidad, más lento en primera carga

```bash
python drecall.py init data.zip
```

### `gemma`
- EmbeddingGemma-300m (ultra-liviano, 300MB)
- Rule-based classifier

```bash
python drecall.py init data.zip --profile gemma
```

## Ejemplos de Uso

### Encontrar conversaciones sobre un tema:

```bash
python drecall.py interactive

> bases de datos vectoriales
# Encuentra: optimización, índices, embeddings, etc.
```

### Buscar persona específica:

```bash
> gustavo
# Hybrid search: prioriza mensajes con "gustavo" literal
```

### Explorar tópicos técnicos:

```bash
> ..python..
# Solo mensajes categorizados con tópico "python"
```

### Ver conversación completa:

```
> react hooks useState
Tab (ir a resultados)
↑↓ (navegar)
Enter (ver conversación)
# Auto-scroll al mensaje matcheado con highlight
```

### Exportar conversación interesante:

```
Enter en resultado → Ver conversación
Ctrl+P → Exportar
# Guardado en: data/exports/export_XXXXX.md
```

## Troubleshooting

### GPU no detectada

```bash
# Forzar CPU
python drecall.py init data.zip --profile lite --cpu-only
```

### Cambiar modelo de vectorización

```bash
# Editar config/lite.yaml
vim config/lite.yaml

# Cambiar:
vectorizer:
  config:
    model: "sentence-transformers/all-MiniLM-L12-v2"

# Re-indexar
python drecall.py reindex
```

### Ver logs de memoria

```bash
cat data/memory_profile.log
```

## Próximas Features

- [ ] Infinite scroll real (implementado, en testing)
- [ ] Integración con AI externas (Gemini, Claude)
- [ ] Indexación de attachments (PDFs, imágenes)
- [ ] HCS completo (Macro/Micro Nodes automáticos)
- [ ] Web UI (vía BintelX)

## Soporte

Documentación completa:
- `README.md` - Arquitectura general
- `ARCHITECTURE.md` - Diseño técnico
- `docs/MODEL_SELECTION.md` - Guía de modelos

¿Preguntas? Abre un issue en el repo.
