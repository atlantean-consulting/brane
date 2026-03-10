# brane Manual

## Overview

brane is a command-line OCR tool that runs entirely on your local machine. It uses the Qwen3-VL vision-language model through Ollama to extract text from images with quality that rivals cloud APIs like Claude and GPT-4V, and the homr neural network engine to recognize sheet music and output MusicXML.

Unlike traditional OCR engines (Tesseract, EasyOCR), brane understands document structure -- headings, lists, tables, code blocks -- and preserves them in the output. For sheet music, it replaces tools like Audiveris with a transformer-based approach that handles printed scores with less manual cleanup.

## Installation

Run the installer:

```bash
./install.sh
```

This will:
1. Install Ollama (if not present)
2. Pull the Qwen3-VL 8B model (~6GB download)
3. Create a Python virtual environment
4. Install brane and its dependencies

To also install the 30B model (~19GB, requires 24GB+ VRAM):

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
| 8b (default) | ~8GB | Fast (~3-8s) | Excellent for most documents |
| 30b | ~22GB | Slower (~10-20s) | Maximum quality, complex layouts |

```bash
brane document.png -m 8b    # default
brane document.png -m 30b   # higher quality
```

The 30B model requires approximately 22-24GB of VRAM. If you have multiple GPUs, Ollama will automatically split the model across them.

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
ollama stop qwen3-vl:8b
```

### Streaming

By default, output streams to the terminal as it's generated. To wait for the complete result:

```bash
brane image.png --no-stream
```

Streaming is automatically disabled when writing to a file (`-o`).

## Sheet Music OCR

brane can recognize printed sheet music and output MusicXML files, which can be opened in MuseScore, Finale, Sibelius, or any MusicXML-compatible editor.

### Basic Usage

```bash
brane music sheet.png
```

This produces `sheet.musicxml` in the same directory.

### PDF Input

Sheet music PDFs are automatically converted to images for processing:

```bash
brane music score.pdf
```

A multi-page PDF produces one MusicXML file per page (e.g., `score_page001.musicxml`, `score_page002.musicxml`).

### Output Options

```bash
# Explicit output file (single input only)
brane music sheet.png -o output.musicxml

# Output to a directory (works with multiple inputs)
brane music *.pdf -o results/

# Higher DPI for PDF rasterization (default: 300)
brane music score.pdf --dpi 600
```

### GPU Acceleration

Sheet music recognition uses ONNX Runtime and will automatically use your NVIDIA GPU if a working CUDA 12 toolkit is installed. If CUDA isn't available, it falls back to CPU with a message. To explicitly disable GPU:

```bash
brane music sheet.png --no-gpu
```

### What It Handles

The homr engine focuses on pitch and rhythm on bass and treble clefs. It handles:

- Printed sheet music (not handwritten)
- Polyphonic scores (multiple voices)
- Standard notation (notes, rests, accidentals, key/time signatures)

It currently has limited support for dynamics, articulation, and double sharps/flats.

### First Run

The first invocation downloads ~130MB of ONNX models. This only happens once.

## Supported Formats

**Text OCR**: PNG, JPEG, WebP, BMP, TIFF, GIF.

**Sheet Music**: PNG, JPEG, PDF.

## Architecture

brane is intentionally minimal:

```
cli.py            # CLI routing (subcommands with backwards-compatible default)
ocr_engine.py     # Text OCR via Ollama/Qwen3-VL
music_engine.py   # Sheet music OMR via homr
prompts.py        # OCR prompt templates
pyproject.toml    # Package metadata
```

### How Text OCR Works

1. You pass an image to `brane`
2. brane sends the image to Ollama's local API (localhost:11434) along with a carefully crafted OCR prompt
3. Qwen3-VL processes the image on your GPU and generates text
4. brane streams the result to stdout or writes it to a file
5. The model unloads from VRAM (unless `--persist` is set)

### How Sheet Music OCR Works

1. You pass an image or PDF to `brane music`
2. PDFs are converted to PNG images via PyMuPDF
3. homr detects staves, noteheads, clefs, and other symbols using a segmentation neural network
4. A transformer model (Polyphonic-TrOMR) recognizes the note sequences
5. The result is written as a MusicXML file

### Prompt Tuning

The text OCR prompts live in `prompts.py`. If you want to adjust brane's default behavior, edit the prompts there. The key principles:

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
ollama pull qwen3-vl:8b
ollama pull qwen3-vl:30b  # optional
```

### Slow first run

The first invocation after a reboot loads the model into VRAM (~5-8 seconds for 8B). Subsequent runs are faster if you use `--persist`. Without `--persist`, every run pays this cost -- but your VRAM stays free between runs.

### Out of VRAM

The 8B model needs ~8GB of VRAM. The 30B model needs ~22GB. If you're running out:

- Close other GPU-heavy applications
- Use the 8B model instead of 30B
- Check what's in VRAM: `nvidia-smi`

### Truncated output on large documents

Very dense, multi-page images may hit the model's context limit. Split large documents into individual pages and process them separately:

```bash
brane page1.png page2.png page3.png -o document.md
```

### Sheet music: "CUDA not fully available, falling back to CPU"

This means onnxruntime can't load the CUDA 12 provider. Sheet music OCR still works on CPU, but GPU is faster. To enable GPU acceleration, install the CUDA 12.8 toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8
```

Then add to your `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```
