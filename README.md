# brane

Local OCR that doesn't suck.

`brane` uses [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) running locally via [Ollama](https://ollama.com) to extract text from images with near-cloud-API quality. No data leaves your machine!

## Quick Start

```bash
# Install everything
./install.sh

# OCR an image
brane screenshot.png

# Save to file
brane document.png -o result.md

# Use the 72B model for maximum quality
brane document.png -m 72b

# OCR multiple images at once
brane *.png -o combined.md

# Custom extraction prompt
brane table.png --prompt "Extract this table as CSV"
```

## Requirements

`brane` is designed for high-horsepower systems, but it should work on any Linux device with a good GPU if you're patient enough. Minimum specs are:

- Linux with NVIDIA GPU (8GB+ VRAM for 7B model, 48GB+ for 72B)
- Python 3.12+
- ~5GB disk space for 7B model, ~48GB for 72B

## Options

```
brane [OPTIONS] IMAGES...

  -o, --output PATH         Write output to file
  -f, --format [markdown|plain]  Output format (default: markdown)
  -m, --model [7b|72b]      Model size (default: 7b)
  -p, --prompt TEXT          Custom prompt override
  --persist                  Keep model in VRAM after completion
  --no-stream                Disable streaming output
  --help                     Show help
```

See [MANUAL.md](MANUAL.md) for detailed documentation.

## Why "brane"?

A [brane](https://en.wikipedia.org/wiki/Brane) is a higher-dimensional membrane in string theory. Tesseract (the OCR engine) is named after a 4D cube. `brane` goes further -- and also happens to describe the planar surfaces that text lives on.
