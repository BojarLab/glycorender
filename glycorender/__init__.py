"""Bespoke SVG renderer for GlycoDraw: SVG in, PDF/PNG/SVG out, with numpy as the only dependency."""
from .render import (convert_chem_to_file, convert_svg_to_pdf, convert_svg_to_png,
                     pdf_to_svg_bytes, simple_svg_to_pdf, simple_svg_to_png)


def __getattr__(name):
    # Read when asked for: importing importlib.metadata took a quarter of the time of importing glycorender
    if name == '__version__':
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("glycorender")
        except PackageNotFoundError:  # running from a source checkout that was never pip-installed
            return "0.0.0+local"
    raise AttributeError(f"module 'glycorender' has no attribute {name!r}")


__all__ = ['convert_chem_to_file', 'convert_svg_to_pdf', 'convert_svg_to_png',
           'pdf_to_svg_bytes', 'simple_svg_to_pdf', 'simple_svg_to_png', '__version__']