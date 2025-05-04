# glycorender

A bespoke SVG to PDF/PNG renderer for the GlycoDraw platform, specialized in accurately rendering glycan structures with support for chemical notations and path-based text positioning.

[![PyPI version](https://img.shields.io/pypi/v/glycorender.svg)](https://pypi.org/project/glycorender/)
[![Python Version](https://img.shields.io/pypi/pyversions/glycorender.svg)](https://pypi.org/project/glycorender/)
[![License](https://img.shields.io/github/license/BojarLab/glycorender.svg)](LICENSE)

## Features

- Accurate SVG to PDF and PNG conversion for glycan structures
- Support for text on paths with proper positioning and orientation
- Special handling for monosaccharide properties like "Df" with proper styling
- Gradient fills for circles and shapes
- Precise connection path rendering between glycan elements
- Support for various SVG path commands

## Installation

```bash
pip install glycorender
```

## Quick Start

```python
from glycorender import convert_svg_to_pdf, convert_svg_to_png

# Convert SVG file to PDF
with open('glycan_structure.svg', 'r') as f:
    svg_data = f.read()
    
convert_svg_to_pdf(svg_data, 'output.pdf')

# Convert SVG file to PNG
convert_svg_to_png(svg_data, 'output.png')

# Or directly from SVG data
svg_data = """..."""
convert_svg_to_pdf(svg_data, 'output.pdf')
convert_svg_to_png(svg_data, 'output.png')
```

## Use Cases

GlycoRender is specifically designed for:

- Converting glycan structure diagrams from GlycoDraw to publication-quality PDFs or PNGs
- Preserving the exact layout and styling of complex carbohydrate representations
- Ensuring chemical notations are properly formatted in the output
- Maintaining the correct connections between monosaccharide units

## API Reference

### `convert_svg_to_pdf(svg_data, pdf_file_path)`

Converts SVG data to a PDF file.

Parameters:
- `svg_data` (str or bytes): SVG content either as a string or bytes
- `pdf_file_path` (str): Path where the output PDF should be saved

### `convert_svg_to_png(svg_data, png_file_path)`

Converts SVG data to a PNG file.

Parameters:
- `svg_data` (str or bytes): SVG content either as a string or bytes
- `png_file_path` (str): Path where the output PNG should be saved

## Supported SVG Elements

- `<path>`: Full support for path commands (M, L, H, V, Z, etc.)
- `<circle>`: Support for basic circles and gradient-filled circles
- `<rect>`: Support for rectangles with stroke and fill
- `<text>` and `<textPath>`: Support for text placement along paths with proper orientation
- `<defs>`: Support for path and gradient definitions
- `<radialGradient>`: Support for radial gradients with stops

## Dependencies

- ReportLab: For PDF generation
- PyMuPDF: For PNG conversion
- Pillow: For PNG metadata injection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.