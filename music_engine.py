"""brane music - Sheet music OCR (OMR) using homr."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import click

_gpu_checked = None

# Known locations for CUDA 12 and cuDNN 9 libraries.
_CUDA_SEARCH_PATHS = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-12.8/lib64",
    "/usr/local/lib/ollama/cuda_v12",       # Ollama's bundled CUDA 12
    "/usr/local/lib/ollama/mlx_cuda_v13",   # Ollama's cuDNN 9
]


def _ensure_cuda_libs_loadable() -> None:
    """Add CUDA/cuDNN library dirs to LD_LIBRARY_PATH if not already present."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    dirs_to_add = [d for d in _CUDA_SEARCH_PATHS if os.path.isdir(d) and d not in ld_path]
    if dirs_to_add:
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs_to_add + ([ld_path] if ld_path else []))


def _cuda_actually_works() -> bool:
    """Test whether onnxruntime can actually create a CUDA session."""
    global _gpu_checked
    if _gpu_checked is not None:
        return _gpu_checked
    _gpu_checked = False
    try:
        _ensure_cuda_libs_loadable()
        import ctypes
        # Pre-load CUDA/cuDNN libs so onnxruntime's provider bridge can find them.
        for lib_name in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12",
                         "libcurand.so.10", "libcufft.so.11", "libcudnn.so.9"):
            for d in _CUDA_SEARCH_PATHS:
                lib_path = os.path.join(d, lib_name)
                if os.path.exists(lib_path):
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    break
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        # Load the CUDA provider bridge — this is what actually fails
        # when CUDA toolkit libs (curand, cublas, cudnn, etc.) are missing.
        lib_dir = os.path.dirname(ort.__file__)
        cuda_lib = os.path.join(lib_dir, "capi", "libonnxruntime_providers_cuda.so")
        if os.path.exists(cuda_lib):
            ctypes.CDLL(cuda_lib)
            _gpu_checked = True
    except OSError:
        pass
    return _gpu_checked

SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}
SUPPORTED_PDF_FORMATS = {".pdf"}
SUPPORTED_FORMATS = SUPPORTED_IMAGE_FORMATS | SUPPORTED_PDF_FORMATS


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[Path]:
    """Convert PDF pages to PNG images in a temp directory."""
    import fitz

    doc = fitz.open(str(pdf_path))
    temp_dir = Path(tempfile.mkdtemp(prefix="brane_music_"))
    images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img_path = temp_dir / f"{pdf_path.stem}_page{i + 1:03d}.png"
        pix.save(str(img_path))
        images.append(img_path)
    doc.close()
    return images


def run_omr(image_path: Path, output_path: Path, use_gpu: bool = True) -> Path:
    """Run homr OMR on a single image, writing MusicXML to output_path."""
    from homr.main import ProcessingConfig, download_weights, process_image
    from homr.music_xml_generator import XmlGeneratorArguments

    use_gpu_final = use_gpu and _cuda_actually_works()
    if use_gpu and not use_gpu_final:
        click.echo("CUDA not fully available, falling back to CPU.", err=True)

    download_weights(use_gpu_final)

    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        use_gpu_inference=use_gpu_final,
    )
    xml_args = XmlGeneratorArguments()

    # homr writes .musicxml next to input, so copy to a temp dir
    work_dir = Path(tempfile.mkdtemp(prefix="brane_omr_"))
    try:
        work_image = work_dir / image_path.name
        shutil.copy2(image_path, work_image)

        process_image(str(work_image), config, xml_args)

        generated = work_image.with_suffix(".musicxml")
        if not generated.exists():
            raise RuntimeError(f"homr did not produce output for {image_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(generated), str(output_path))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_path


@click.command("music")
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path),
              help="Output file or directory. Defaults to input name with .musicxml extension.")
@click.option("--dpi", default=300, help="DPI for PDF rasterization (default: 300).")
@click.option("--no-gpu", is_flag=True, help="Disable GPU acceleration.")
def music(inputs, output, dpi, no_gpu):
    """Recognize sheet music and output MusicXML.

    Accepts images (PNG, JPG) and PDF files.

    Examples:

        brane music sheet.png

        brane music score.pdf -o score.musicxml

        brane music page1.png page2.png -o output_dir/
    """
    validated = []
    for inp in inputs:
        if inp.suffix.lower() not in SUPPORTED_FORMATS:
            click.echo(
                f"Error: Unsupported format: {inp.suffix} "
                f"(supported: {', '.join(sorted(SUPPORTED_FORMATS))})",
                err=True,
            )
            sys.exit(1)
        validated.append(inp)

    # Expand PDFs to images, track temp dirs for cleanup
    all_images = []  # (image_path, output_stem)
    cleanup_dirs = set()
    for inp in validated:
        if inp.suffix.lower() in SUPPORTED_PDF_FORMATS:
            click.echo(f"Converting PDF: {inp.name}", err=True)
            page_images = pdf_to_images(inp, dpi=dpi)
            cleanup_dirs.add(page_images[0].parent)
            for img in page_images:
                all_images.append((img, img.stem))
        else:
            all_images.append((inp, inp.stem))

    # Determine output paths
    if output and output.suffix == ".musicxml":
        if len(all_images) > 1:
            click.echo(
                "Error: Cannot use a single output file with multiple inputs. "
                "Use a directory instead.",
                err=True,
            )
            sys.exit(1)
        output_paths = [output]
    elif output:
        output.mkdir(parents=True, exist_ok=True)
        output_paths = [output / f"{stem}.musicxml" for _, stem in all_images]
    else:
        base_dir = validated[0].parent
        output_paths = [base_dir / f"{stem}.musicxml" for _, stem in all_images]

    # Process each image
    try:
        for (img_path, _stem), out_path in zip(all_images, output_paths):
            click.echo(f"Processing: {img_path.name}", err=True)
            try:
                run_omr(img_path, out_path, use_gpu=not no_gpu)
                click.echo(f"Written: {out_path}", err=True)
            except Exception as e:
                click.echo(f"Error processing {img_path.name}: {e}", err=True)
                if len(all_images) == 1:
                    sys.exit(1)
    finally:
        for d in cleanup_dirs:
            shutil.rmtree(d, ignore_errors=True)

    click.echo(f"Done. Processed {len(all_images)} image(s).", err=True)
