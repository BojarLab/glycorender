"""Bespoke SVG renderer for GlycoDraw: SVG in, PDF/PNG/SVG out, with numpy as the only dependency."""
from .render import (convert_chem_to_file, convert_svg_to_pdf, convert_svg_to_png,
                     pdf_to_svg_bytes, simple_svg_to_pdf, simple_svg_to_png)

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("glycorender")
except PackageNotFoundError:  # running from a source checkout that was never pip-installed
    __version__ = "0.0.0+local"

__all__ = ['convert_chem_to_file', 'convert_svg_to_pdf', 'convert_svg_to_png',
           'pdf_to_svg_bytes', 'simple_svg_to_pdf', 'simple_svg_to_png', '__version__']