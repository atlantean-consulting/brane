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

def write_musicxml(tree_or_path, output_path: Path) -> None:
    """Write a MusicXML file as .musicxml (plain XML) or .mxl (compressed)."""
    import xml.etree.ElementTree as ET
    import zipfile

    if isinstance(tree_or_path, (str, Path)):
        tree = ET.parse(tree_or_path)
    else:
        tree = tree_or_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".mxl":
        # .mxl is a ZIP archive containing the MusicXML + container manifest
        xml_bytes = ET.tostring(tree.getroot(), encoding="UTF-8", xml_declaration=True)
        container = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<container>\n'
            '  <rootfiles>\n'
            '    <rootfile full-path="score.musicxml"/>\n'
            '  </rootfiles>\n'
            '</container>\n'
        )
        with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("META-INF/container.xml", container)
            zf.writestr("score.musicxml", xml_bytes)
    else:
        ET.indent(tree)
        tree.write(str(output_path), encoding="UTF-8", xml_declaration=True)


SUPPORTED_OUTPUT_FORMATS = {".musicxml", ".mxl"}
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


def clean_image(image_path: Path) -> Path:
    """Binarize and clean a scanned sheet music image for better OMR."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adaptive threshold to remove uneven background (yellowed pages, shadows)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15
    )

    # Small morphological opening to remove noise specks
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Remove dark scan borders (binding shadow, scanner edge)
    for col in range(w // 5):
        if np.mean(cleaned[:, col]) > 230:
            break
        cleaned[:, col] = 255
    for col in range(w - 1, w - w // 5, -1):
        if np.mean(cleaned[:, col]) > 230:
            break
        cleaned[:, col] = 255
    for row in range(h // 10):
        if np.mean(cleaned[row, :]) > 230:
            break
        cleaned[row, :] = 255
    for row in range(h - 1, h - h // 10, -1):
        if np.mean(cleaned[row, :]) > 230:
            break
        cleaned[row, :] = 255

    out_path = image_path.with_stem(image_path.stem + "_clean")
    cv2.imwrite(str(out_path), cleaned)
    return out_path


def run_omr(image_path: Path, output_path: Path, use_gpu: bool = True,
            clean: bool = False) -> Path:
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

        if clean:
            work_image = clean_image(work_image)

        process_image(str(work_image), config, xml_args)

        generated = work_image.with_suffix(".musicxml")
        if not generated.exists():
            raise RuntimeError(f"homr did not produce output for {image_path.name}")

        if output_path.suffix.lower() == ".mxl":
            write_musicxml(generated, output_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(generated), str(output_path))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_path


def concat_musicxml(files: list[Path], output_path: Path) -> None:
    """Concatenate multiple MusicXML files into one by appending measures."""
    import xml.etree.ElementTree as ET

    if not files:
        return

    base_tree = ET.parse(files[0])
    base_root = base_tree.getroot()

    # Find all <part> elements in the base file and index by id
    base_parts = {p.get("id"): p for p in base_root.findall(".//part")}

    for extra_file in files[1:]:
        extra_tree = ET.parse(extra_file)
        extra_root = extra_tree.getroot()

        for extra_part in extra_root.findall(".//part"):
            part_id = extra_part.get("id")
            if part_id not in base_parts:
                continue
            base_part = base_parts[part_id]

            # Renumber measures to continue from the base
            existing = base_part.findall("measure")
            next_num = max((int(m.get("number", 0)) for m in existing), default=0) + 1

            for measure in extra_part.findall("measure"):
                measure.set("number", str(next_num))
                next_num += 1
                base_part.append(measure)

    write_musicxml(base_tree, output_path)


@click.command("music")
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path),
              help="Output file or directory. Defaults to input name with .musicxml extension.")
@click.option("--dpi", default=300, help="DPI for PDF rasterization (default: 300).")
@click.option("--no-gpu", is_flag=True, help="Disable GPU acceleration.")
@click.option("-c", "--concat", is_flag=True,
              help="Concatenate all pages into a single MusicXML file.")
@click.option("--clean", is_flag=True,
              help="Pre-process images (binarize, remove borders) for scanned/old scores.")
def music(inputs, output, dpi, no_gpu, concat, clean):
    """Recognize sheet music and output MusicXML.

    Accepts images (PNG, JPG) and PDF files.

    Examples:

        brane music sheet.png

        brane music score.pdf -o score.musicxml

        brane music page1.png page2.png -o output_dir/

        brane music pg1.pdf pg2.pdf -c -o full_score.musicxml

        brane music old_scan.pdf --clean -o result.musicxml
    """
    if concat and not output:
        click.echo("Error: --concat requires -o to specify the output file.", err=True)
        sys.exit(1)

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

    # When concatenating, process into a temp dir then merge
    if concat:
        concat_dir = Path(tempfile.mkdtemp(prefix="brane_concat_"))
        cleanup_dirs.add(concat_dir)
        output_paths = [concat_dir / f"{stem}.musicxml" for _, stem in all_images]
    elif output and output.suffix.lower() in SUPPORTED_OUTPUT_FORMATS:
        if len(all_images) > 1:
            click.echo(
                "Error: Cannot use a single output file with multiple inputs. "
                "Use -c/--concat to merge, or specify a directory.",
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
    completed = []
    try:
        for (img_path, _stem), out_path in zip(all_images, output_paths):
            click.echo(f"Processing: {img_path.name}", err=True)
            try:
                run_omr(img_path, out_path, use_gpu=not no_gpu, clean=clean)
                completed.append(out_path)
                if not concat:
                    click.echo(f"Written: {out_path}", err=True)
            except Exception as e:
                click.echo(f"Error processing {img_path.name}: {e}", err=True)
                if len(all_images) == 1:
                    sys.exit(1)

        if concat and completed:
            click.echo(f"Concatenating {len(completed)} page(s)...", err=True)
            concat_musicxml(completed, output)
            click.echo(f"Written: {output}", err=True)
    finally:
        for d in cleanup_dirs:
            shutil.rmtree(d, ignore_errors=True)

    click.echo(f"Done. Processed {len(all_images)} image(s).", err=True)
