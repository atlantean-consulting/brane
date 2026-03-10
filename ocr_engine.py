"""brane - Local VLM-based OCR using Qwen3-VL via Ollama."""

import sys
from pathlib import Path

import click
import ollama

from prompts import MARKDOWN_PROMPT, PLAIN_PROMPT

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
MODELS = {
    "8b": "qwen3-vl:8b",
    "30b": "qwen3-vl:30b",
}
DEFAULT_MODEL = "8b"


def validate_image(path: Path) -> Path:
    if not path.exists():
        raise click.BadParameter(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise click.BadParameter(
            f"Unsupported format: {path.suffix} (supported: {', '.join(sorted(SUPPORTED_FORMATS))})"
        )
    return path


def ocr_image(image_path: Path, model: str, prompt: str, stream: bool = True, keep_alive: int = 0):
    """Run OCR on a single image and yield text chunks."""
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        stream=stream,
        keep_alive=keep_alive,
    )

    if stream:
        for chunk in response:
            text = chunk["message"]["content"]
            if text:
                yield text
    else:
        yield response["message"]["content"]


@click.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Write output to file.")
@click.option(
    "-f", "--format", "fmt", type=click.Choice(["markdown", "plain"]), default="markdown",
    help="Output format (default: markdown).",
)
@click.option(
    "-m", "--model", "model_size", type=click.Choice(list(MODELS.keys())), default=DEFAULT_MODEL,
    help=f"Model size (default: {DEFAULT_MODEL}).",
)
@click.option("-p", "--prompt", "custom_prompt", default=None, help="Custom prompt override.")
@click.option("--no-stream", is_flag=True, help="Disable streaming output.")
@click.option("--persist", is_flag=True, help="Keep model loaded in VRAM after completion.")
def main(images, output, fmt, model_size, custom_prompt, no_stream, persist):
    """brane - Extract text from images using local VLM OCR.

    Examples:

        brane screenshot.png

        brane document.png -o result.md

        brane *.png -m 30b -o combined.md

        brane photo.jpg --format plain

        brane table.png --prompt "Extract this table as CSV"
    """
    model = MODELS[model_size]
    prompt = custom_prompt or (MARKDOWN_PROMPT if fmt == "markdown" else PLAIN_PROMPT)

    validated = []
    for img in images:
        try:
            validated.append(validate_image(img))
        except click.BadParameter as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    try:
        ollama.list()
    except Exception:
        click.echo(
            "Error: Cannot connect to Ollama. Is it running?\n"
            "  Start it with: ollama serve",
            err=True,
        )
        sys.exit(1)

    output_parts = []
    for i, img_path in enumerate(validated):
        if len(validated) > 1:
            header = f"\n--- {img_path.name} ---\n\n" if i > 0 else f"--- {img_path.name} ---\n\n"
            if output:
                output_parts.append(header)
            else:
                click.echo(header, nl=False)

        keep_alive = -1 if persist else 0
        if output:
            for chunk in ocr_image(img_path, model, prompt, stream=False, keep_alive=keep_alive):
                output_parts.append(chunk)
            if i < len(validated) - 1:
                output_parts.append("\n\n")
        else:
            for chunk in ocr_image(img_path, model, prompt, stream=not no_stream, keep_alive=keep_alive):
                click.echo(chunk, nl=False)
            click.echo()

    if output:
        output.write_text("".join(output_parts))
        click.echo(f"Written to {output}", err=True)


if __name__ == "__main__":
    main()
