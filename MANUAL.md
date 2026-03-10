# brane Manual

## Overview

brane is a command-line OCR tool that runs entirely on your local machine. It uses the Qwen2.5-VL vision-language model through Ollama to extract text from images with quality that rivals cloud APIs like Claude and GPT-4V.

Unlike traditional OCR engines (Tesseract, EasyOCR), brane understands document structure -- headings, lists, tables, code blocks -- and preserves them in the output.

## Installation

Run the installer:

```bash
./install.sh
```

This will:
1. Install Ollama (if not present)
2. Pull the Qwen2.5-VL 7B model (~5GB download)
3. Create a Python virtual environment
4. Install brane and its dependencies

To also install the 72B model (~48GB, requires 48GB+ VRAM):

```bash
./install.sh --big
```

## Usage

### Basic OCR

```bash
brane image.png
```

Text streams to stdout in markdown format as it's generated.

### Save to File

```bash
brane image.png -o output.md
```

When writing to a file, streaming is disabled and the complete result is written at once.

### Multiple Images

```bash
brane page1.png page2.png page3.png -o full-document.md
```

Or with a glob:

```bash
brane *.png -o combined.md
```

Each image is separated by a `--- filename ---` header in the output.

### Output Formats

**Markdown** (default): Preserves document structure with headings, lists, tables, and code blocks.

```bash
brane document.png
brane document.png --format markdown  # same thing
```

**Plain text**: Raw text with original line breaks, no markdown formatting.

```bash
brane document.png --format plain
```

### Model Selection

brane ships with two model sizes:

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| 7b (default) | ~5GB | Fast (~3-8s) | Excellent for most documents |
| 72b | ~48GB | Slower (~15-30s) | Maximum quality, complex layouts |

```bash
brane document.png -m 7b    # default
brane document.png -m 72b   # higher quality
```

The 72B model requires approximately 48GB of VRAM. If you have multiple GPUs, Ollama will automatically shard the model across them.

### Custom Prompts

Override the default OCR prompt for specialized extraction:

```bash
# Extract a table as CSV
brane table.png --prompt "Extract this table as CSV with headers"

# Get just the title
brane cover.png --prompt "What is the title of this document?"

# Structured extraction
brane invoice.png --prompt "Extract: date, invoice number, total amount, and line items as JSON"
```

### VRAM Management

By default, brane unloads the model from VRAM immediately after each run. This is ideal for one-off OCR tasks where you don't want a multi-gigabyte model sitting in your GPU memory.

If you're processing many files in a session, use `--persist` to keep the model loaded between runs:

```bash
brane page1.png --persist
brane page2.png --persist   # much faster, model already in VRAM
brane page3.png             # last one, let it unload
```

To manually unload a model at any time:

```bash
ollama stop qwen2.5vl:7b
```

### Streaming

By default, output streams to the terminal as it's generated. To wait for the complete result:

```bash
brane image.png --no-stream
```

Streaming is automatically disabled when writing to a file (`-o`).

## Supported Image Formats

PNG, JPEG, WebP, BMP, TIFF, GIF.

## Architecture

brane is intentionally minimal:

```
ocr_engine.py   # CLI and Ollama interaction (~130 lines)
prompts.py      # OCR prompt templates
pyproject.toml  # Package metadata
```

It's a thin wrapper around the Ollama Python client. The intelligence comes from Qwen2.5-VL, not from brane's code.

### How It Works

1. You pass an image to `brane`
2. brane sends the image to Ollama's local API (localhost:11434) along with a carefully crafted OCR prompt
3. Qwen2.5-VL processes the image on your GPU and generates text
4. brane streams the result to stdout or writes it to a file
5. The model unloads from VRAM (unless `--persist` is set)

### Prompt Tuning

The OCR prompts live in `prompts.py`. If you want to adjust brane's default behavior, edit the prompts there. The key principles:

- Tell the model it's "a precise OCR engine" (prevents it from describing the image)
- Say "output ONLY the extracted text" (prevents chatty preambles)
- Specify the formatting rules explicitly (markdown structure, tables, code blocks)

## Troubleshooting

### "Cannot connect to Ollama"

Ollama isn't running. Start it:

```bash
ollama serve
```

Or if it's installed as a systemd service:

```bash
sudo systemctl start ollama
```

### Model not found

Pull the model:

```bash
ollama pull qwen2.5vl:7b
ollama pull qwen2.5vl:72b  # optional
```

### Slow first run

The first invocation after a reboot loads the model into VRAM (~5-8 seconds for 7B). Subsequent runs are faster if you use `--persist`. Without `--persist`, every run pays this cost -- but your VRAM stays free between runs.

### Out of VRAM

The 7B model needs ~5GB of VRAM. The 72B model needs ~48GB. If you're running out:

- Close other GPU-heavy applications
- Use the 7B model instead of 72B
- Check what's in VRAM: `nvidia-smi`

### Truncated output on large documents

Very dense, multi-page images may hit the model's context limit. Split large documents into individual pages and process them separately:

```bash
brane page1.png page2.png page3.png -o document.md
```
