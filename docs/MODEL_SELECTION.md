# Guía de Selección de Modelos

## Clasificadores (Intent Classification)

### Métricas Clave a Buscar en HuggingFace:

1. **Parameters** (Parámetros del modelo)
   - 🎯 Objetivo: **<100M** parámetros
   - ✅ Óptimo: 20M-80M (distil*, deberta-v3-xsmall)
   - ⚠️ Evitar: >200M (muy lento en GPU <8GB)

2. **Model Size** (Tamaño en disco)
   - 🎯 Objetivo: **<500MB**
   - ✅ Óptimo: 200-400MB
   - ⚠️ Evitar: >1GB (carga lenta, ocupa mucha VRAM)

3. **VRAM Usage** (Uso de memoria GPU)
   - 🎯 Objetivo: **<1GB**
   - ✅ Óptimo: 400-800MB
   - ⚠️ Evitar: >2GB (incompatible con pipeline serial)

4. **Inference Speed** (Velocidad de inferencia)
   - 🎯 Objetivo: **>50 samples/sec** en GPU
   - ✅ Óptimo: 100-500 samples/sec
   - Buscar en model card: "throughput" o "samples/sec"

5. **Task Type**
   - ✅ Usar: `zero-shot-classification` o `text-classification`
   - ✅ Alternativa: `nli` (Natural Language Inference)
   - ❌ Evitar: `question-answering` (diferente propósito)

### Modelos Recomendados (Enero 2025)

#### Ultra-Ligero (GPU <4GB):
```yaml
classifier:
  class: "src.classifiers.transformer.TransformerClassifier"
  config:
    model: "prajjwal1/bert-tiny"
    # 4.4M params, ~200MB VRAM, 500+ samples/sec
```

#### Ligero (GPU 4-6GB):
```yaml
classifier:
  class: "src.classifiers.transformer.TransformerClassifier"
  config:
    model: "cross-encoder/nli-deberta-v3-xsmall"
    # 22M params, ~400MB VRAM, 200+ samples/sec
```

#### Balanceado (GPU 6-8GB) - **RECOMENDADO**:
```yaml
classifier:
  class: "src.classifiers.transformer.TransformerClassifier"
  config:
    model: "cross-encoder/nli-distilroberta-base"
    # 82M params, ~800MB VRAM, 100+ samples/sec
    # Excelente balance calidad/velocidad
```

#### Full (GPU >8GB):
```yaml
classifier:
  class: "src.classifiers.transformer.TransformerClassifier"
  config:
    model: "valhalla/distilbart-mnli-12-1"
    # 139M params, ~1.2GB VRAM, 50+ samples/sec
    # Mejor calidad pero más lento
```

---

## Vectorizers (Embeddings)

### Métricas Clave:

1. **Embedding Dimension**
   - 🎯 Objetivo: **384** (balance perfecto)
   - ✅ Alternativa: 768 (más calidad, más VRAM)
   - ⚠️ Evitar: >1024 (overkill para chat history)

2. **Model Size**
   - 🎯 Objetivo: **<500MB**
   - ✅ Óptimo: 100-400MB
   - ⚠️ Evitar: >1GB

3. **VRAM Usage**
   - 🎯 Objetivo: **<1GB**
   - ✅ Óptimo: 400-800MB
   - Para calcular: ~= Model Size × 1.5

4. **MTEB Score** (Massive Text Embedding Benchmark)
   - 🎯 Objetivo: **>55**
   - ✅ Óptimo: 58-65 (excelente para chat history)
   - 🏆 Top tier: >65 (para uso profesional)
   - Buscar en: https://huggingface.co/spaces/mteb/leaderboard

5. **Multilingual Support** (si necesitas español)
   - ✅ Buscar: "multilingual" o "multilang" en el nombre
   - ✅ Verificar: languages en model card

### Modelos Recomendados (Enero 2025)

#### Ultra-Ligero (GPU <4GB):
```yaml
vectorizer:
  class: "src.vectorizers.sentence_transformer.SentenceTransformerVectorizer"
  config:
    model: "sentence-transformers/paraphrase-MiniLM-L3-v2"
    # Dim: 384, ~120MB VRAM, MTEB: 50.2
```

#### Ligero (GPU 4-6GB) - **ACTUALMENTE USAS ESTE** ✅:
```yaml
vectorizer:
  class: "src.vectorizers.sentence_transformer.SentenceTransformerVectorizer"
  config:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    # Dim: 384, ~400MB VRAM, MTEB: 58.8
    # El mejor balance calidad/velocidad
```

#### Balanceado (GPU 6-8GB):
```yaml
vectorizer:
  class: "src.vectorizers.sentence_transformer.SentenceTransformerVectorizer"
  config:
    model: "BAAI/bge-small-en-v1.5"
    # Dim: 384, ~600MB VRAM, MTEB: 62.1
    # Mejor calidad que MiniLM
```

#### Multilingüe (Español + English):
```yaml
vectorizer:
  class: "src.vectorizers.sentence_transformer.SentenceTransformerVectorizer"
  config:
    model: "intfloat/multilingual-e5-small"
    # Dim: 384, ~500MB VRAM, MTEB: 60.9
    # Excelente para ES+EN
```

#### Full (GPU >8GB):
```yaml
vectorizer:
  class: "src.vectorizers.sentence_transformer.SentenceTransformerVectorizer"
  config:
    model: "BAAI/bge-base-en-v1.5"
    # Dim: 768, ~1.1GB VRAM, MTEB: 63.6
```

---

## Cómo Evaluar en HuggingFace

### 1. Buscar en HuggingFace:
```
https://huggingface.co/models?pipeline_tag=sentence-similarity
Filter: Task = "Sentence Similarity"
Sort by: Downloads (más populares = más testeados)
```

### 2. Revisar Model Card:
- **"Model Details" section**: Buscar "Parameters" o "Size"
- **"Evaluation" section**: Buscar MTEB scores
- **"Environmental Impact" section**: A veces lista VRAM

### 3. Verificar en Papers/Blogs:
```
https://huggingface.co/spaces/mteb/leaderboard
```
Filtrar por:
- "Model Size" < 500MB
- "Avg" score > 55

### 4. Testear antes de producción:
```bash
# Probar con subset de 1000 mensajes
python drecall.py reindex --limit 1000

# Si funciona bien, hacer full reindex
python drecall.py reindex
```

---

## Recomendación para tu RTX 2080 Super (8GB)

**Setup Óptimo:**

```yaml
# config/optimized.yaml
components:
  classifier:
    class: "src.classifiers.transformer.TransformerClassifier"
    config:
      model: "cross-encoder/nli-distilroberta-base"
      # 82M params, 800MB VRAM, rápido
      batch_size: 32
      device: "cuda"

  vectorizer:
    class: "src.vectorizers.sentence_transformer.SentenceTransformerVectorizer"
    config:
      model: "BAAI/bge-small-en-v1.5"
      # Mejor que MiniLM, solo 200MB más VRAM
      batch_size: 128
      device: "cuda"
```

**Uso de VRAM:**
- Classifier: ~800MB
- Vectorizer: ~600MB
- **Total secuencial**: <1.5GB (perfecto para 8GB)

---

## Cuándo Usar Cada Profile

### `default.yaml` - Máxima Calidad
- GPU >8GB o paciencia para esperar
- Transformer classifier + vectorizer
- Primera carga lenta, mejor accuracy

### `lite.yaml` - Balance ⭐ **RECOMENDADO**
- Cualquier GPU o CPU
- Rule-based classifier (instantáneo) + vectorizer en GPU
- **Carga en <1 minuto, clasificación funcional**

### `express.yaml` - Ultra-Rápido
- Solo vectorización, sin classifier
- Ideal para datasets gigantes (>100k mensajes)
- Agregar classifier después: `drecall reindex`

---

## Switching Models (Sin Reimportar)

```bash
# 1. Cambiar modelo en config
vim config/default.yaml

# 2. Re-vectorizar (no reimporta, solo actualiza vectors)
python drecall.py reindex

# 3. Probar búsqueda
python drecall.py search "test query"
```

El sistema trackea qué vectorizer usaste en cada mensaje y solo re-vectoriza los necesarios.
