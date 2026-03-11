# Known Issues

## Sheet Music OCR Quality

### Source material matters enormously

The homr OMR engine is trained on clean, modern, computer-engraved sheet music. Recognition quality degrades significantly with:

- **Old typeset music** (pre-digital era engraving)
- **Handwritten or hand-copied scores**
- **Scanned book facsimiles** with binding shadows, yellowed pages, or bleed-through
- **Scores with lyrics between staves** (text can confuse staff detection)
- **Low-resolution or noisy scans**

This is not specific to brane -- Audiveris, oemer, and other OMR tools struggle with the same material for the same reasons. OMR as a field is still largely optimized for clean modern notation. We encourage development in the field, so that more older music can be brought into the digital world.

### Case study: 1940s Socialist songbook

We tested `brane` on [Socialist and Labor Songs](https://pmpress.org/index.php?l=product_detail&p=571), a facsimile edition of a collection from the mid-20th century. We scanned the book at 1200 DPI on a flatbed scanner designed for the purpose. We picked "The Internationale" as our first test because it's a classic song with a nice tune.

The pages featured:

- Old movable-type music engraving (not modern computer typesetting)
- Four verses of lyrics interleaved between treble and bass clef staves
- Variable ink density
- Scanner artifacts on page edges

**Results without `--clean`**: homr detected the correct number of staves and produced structurally valid MusicXML, but many notes were wrong. The pitches were mostly accurate, but the rhythms weren't. The result was recognizable as "The Internationale", but sounded like it was being played by a brain-addled capitalist.

**Results with `--clean`**: The adaptive binarization actually made recognition *worse*. The aggressive thresholding altered note head shapes and stem thickness in ways that confused homr's neural network, which was trained on grayscale images with natural contrast gradients. The cleaned images looked sharper to the human eye but were harder for the model to interpret. The result was entirely half notes. When played, it was not recognizable as "The Internationale".

**Takeaway**: `--clean` is not universally beneficial. It helps most with very low contrast scans where the background and foreground are hard to distinguish. For material where the notation is already reasonably visible, the raw image may give better results. Always try both and compare.

### What works well

For best results, use `brane music` with:

- PDF exports from notation software (MuseScore, Finale, Sibelius, LilyPond)
- Clean scans of modern printed sheet music (post-1980s engraving)
- High-resolution images (600+ DPI for scans)
- Scores without lyrics or with lyrics below the grand staff (not between staves)
- Standard Western notation on treble and bass clefs

## Output Format Confusion

`.musicxml` and `.mxl` are **not** interchangeable file extensions (unlike `.jpg`/`.jpeg`)!

- **`.musicxml`** is plain XML text
- **`.mxl`** is a compressed ZIP archive containing MusicXML + a container manifest

MuseScore and other editors will reject a plain XML file saved with a `.mxl` extension. Use the correct extension for the format you want -- `brane` handles both automatically based on the extension you provide with `-o`.

## CUDA / GPU Acceleration

### ONNX Runtime warnings about node assignment

Messages like `Some nodes were not assigned to the preferred execution providers` are normal onnxruntime warnings, not errors. ONNX Runtime intentionally moves lightweight operations (shape calculations, *etc.*) to CPU even when GPU is available. This improves performance, even though some of the errors warn about "the operation being very slow". These warnings do not indicate a problem.

### CUDA version requirements

onnxruntime's CUDA provider requires CUDA 12.x and cuDNN 9.x. Systems with only CUDA 13.x installed will fall back to CPU. brane auto-discovers CUDA libraries from standard system paths and Ollama's bundled libraries, but the full CUDA 12 toolkit is needed for GPU acceleration. See MANUAL.md for installation instructions.

### NVIDIA apt repo conflicts

Adding NVIDIA's CUDA repository can cause package conflicts with existing driver packages (*e.g.*, `libnvidia-gl` conflicts with `libnvidia-egl-gbm1`) because the repo has a high default pin priority. See MANUAL.md for the apt pinning fix.
