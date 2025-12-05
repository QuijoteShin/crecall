# Chat Recall 🧠
> **Tu "Segunda Memoria" para ChatGPT**

¿Alguna vez has intentado buscar algo que hablaste con ChatGPT hace 6 meses y el buscador oficial no encuentra nada? **Chat Recall** soluciona eso.

Es una herramienta de **Deep Search Offline** que indexa todo tu historial de conversaciones y te permite buscar por **significado**, no solo por palabras clave.

![Demo](https://via.placeholder.com/800x400?text=Chat+Recall+TUI+Demo)

## ¿Por qué usar esto?

- 🔍 **Búsqueda Semántica Real**: Encuentra "código de python para api" aunque hayas escrito "script de flask para backend".
- 🔒 **100% Privado y Offline**: Tus datos nunca salen de tu PC. Todo corre localmente.
- 🚀 **Vector First**: Prioriza el *concepto* sobre la palabra exacta.
- 📂 **Organización Automática**: Detecta cuando cambias de tema dentro de un mismo chat y lo segmenta (HCS).
- ⚡ **Interfaz Hacker (TUI)**: Navega tus chats como un pro desde la terminal.

## Instalación Rápida

Requisitos: Python 3.10+ (y opcionalmente una GPU NVIDIA para ir volando, pero funciona en CPU).

```bash
# 1. Clona el repo
git clone https://github.com/tu-usuario/chatrecall.git
cd chatrecall

# 2. Prepara el entorno
python3 -m venv .venv
source .venv/bin/activate  # O en Windows: .venv\Scripts\activate

# 3. Instala
pip install -e .
```

## Cómo Usar

### 1. Exporta tus datos de ChatGPT
Ve a ChatGPT -> Settings -> Data Controls -> Export Data. Recibirás un `.zip`.

### 2. Indexa tu historial
```bash
# Reemplaza con la ruta a tu zip
python crec.py init ruta/a/tu/export.zip --profile express
```
*Tip: `--profile express` es ideal para empezar rápido.*

### 3. ¡Busca!
Lanza la interfaz interactiva:
```bash
python crec.py interactive
```

## Trucos de Búsqueda

| Sintaxis | Qué hace | Ejemplo |
|----------|----------|---------|
| `texto normal` | Búsqueda semántica (por significado) | `receta de pasta` |
| `"texto exacto"` | Búsqueda exacta (Ctrl+F clásico) | `"def __init__"` |
| `..tema..` | Filtra por contexto/tema específico | `..python.. error de import` |

## Preguntas Frecuentes

**¿Necesito una GPU potente?**
No. El modo `--profile express` usa modelos ligeros que corren bien en CPU. Si tienes GPU, úsala para indexar más rápido.

**¿Mis datos se envían a alguna nube?**
No. Absolutamente todo (base de datos, vectores, modelos) vive en tu carpeta local.

**¿Soporta otros formatos?**
Actualmente ChatGPT (.zip). Pronto: Claude y WhatsApp.

---
*Hecho con ❤️ para los que hablamos demasiado con la IA.*
