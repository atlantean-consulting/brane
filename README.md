# brane

Local OCR that doesn't suck.

`brane` uses [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) running locally via [Ollama](https://ollama.com) to extract text from images, and [homr](https://github.com/liebharc/homr) to recognize sheet music. No data leaves your machine!

## Quick Start

```bash
# Install everything
./install.sh

# OCR an image
brane screenshot.png

# Save to file
brane document.png -o result.md

# Use the 30B model for maximum quality
brane document.png -m 30b

# OCR multiple images at once
brane *.png -o combined.md

# Custom extraction prompt
brane table.png --prompt "Extract this table as CSV"

# Sheet music OCR (outputs MusicXML)
brane music sheet.png
brane music score.pdf -o score.musicxml
```

## Requirements

`brane` is designed for high-horsepower systems, but it should work on any Linux device with a good GPU if you're patient enough. Minimum specs are:

- Linux with NVIDIA GPU (8GB+ VRAM for 8B model, 24GB+ for 30B)
- Python 3.12+
- ~6GB disk space for 8B model, ~19GB for 30B

## Text OCR Options

```
brane [OPTIONS] IMAGES...

  -o, --output PATH         Write output to file
  -f, --format [markdown|plain]  Output format (default: markdown)
  -m, --model [8b|30b]      Model size (default: 8b)
  -p, --prompt TEXT          Custom prompt override
  --persist                  Keep model in VRAM after completion
  --no-stream                Disable streaming output
  --help                     Show help
```

## Sheet Music Options

```
brane music [OPTIONS] INPUTS...

  -o, --output PATH    Output file or directory (default: <input>.musicxml)
  --dpi INTEGER        DPI for PDF rasterization (default: 300)
  --no-gpu             Disable GPU acceleration
  --help               Show help
```

See [MANUAL.md](MANUAL.md) for detailed documentation.

## Why "brane"?

A [brane](https://en.wikipedia.org/wiki/Brane) is a higher-dimensional membrane in string theory. Tesseract (the OCR engine) is named after a 4D cube. `brane` goes further -- and also happens to describe the planar surfaces that text lives on.
