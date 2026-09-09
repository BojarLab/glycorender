"""Tests for glycorender: the SVG subset GlycoDraw emits, and the PNG/PDF/SVG backends."""
import zlib
import numpy as np
import pytest
from glycorender import pdfwrite, raster
from glycorender.render import (_render_svg_to_pdf_canvas, convert_svg_to_pdf, convert_svg_to_png, parse_color,
                                parse_path, pdf_to_svg_bytes)

SNFG_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="200" height="110" viewBox="-150.0 -55.0 200 110">
<defs>
<path d="M-100,0 L0,0" stroke-width="4.0" stroke="#000000" id="d0" />
</defs>
<g transform="rotate(0 -50.0 0.0)">
<use xlink:href="#d0" />
<text font-size="20.0" text-anchor="middle" fill="#000000" valign="middle"><textPath xlink:href="#d0" startOffset="50%">
<tspan dy="-0.5em">b 3</tspan>
</textPath></text>
<rect x="-25.0" y="-25.0" width="50" height="50" fill="#FCC326" stroke-width="2.0" stroke="#000000" />
<circle cx="-100" cy="0" r="25.0" fill="#0090BC" stroke-width="2.0" stroke="#000000" />
</g>
</svg>"""


def _inflate(blob):
    """A PDF stream body, decompressed when it is FlateDecode and skipped when it is not."""
    try:
        return zlib.decompress(blob)
    except zlib.error:
        return b''


def _canvas(**effects):
    """A rendered display list for SNFG_SVG, with shadow/sticker set the way the public entry points set them."""
    c = _render_svg_to_pdf_canvas(SNFG_SVG, None, alt_text_info = None)
    for k, v in effects.items():
        setattr(c, k, v)
    return c


def _pixels(c, scale = 3.0):
    """Rasterize a canvas the way to_png does, but keep the RGBA array instead of PNG bytes."""
    return raster.render(c.ops, c.width, c.height, scale, scale, None, *c._effects())


def test_parse_color_never_returns_pure_black():
    assert parse_color('#000000') == pytest.approx((0.110, 0.098, 0.090), abs = 1e-3)
    assert parse_color('#FCC326') == pytest.approx((252/255, 195/255, 38/255), abs = 1e-3)
    assert parse_color('none') is None


def test_parse_path_reads_commands():
    assert parse_path("M-100,0 L0,0") == [('M', [-100.0, 0.0]), ('L', [0.0, 0.0])]


def test_png_header_matches_requested_width():
    png = convert_svg_to_png(SNFG_SVG, None, output_width = 240, return_bytes = True)
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    assert int.from_bytes(png[16:20], 'big') == 240


def test_pdf_is_written_with_subset_font(tmp_path):
    out = tmp_path / "x.pdf"
    convert_svg_to_pdf(SNFG_SVG, str(out))
    data = out.read_bytes()
    assert data[:4] == b'%PDF' and b'/FontFile2' in data and data.rstrip().endswith(b'%%EOF')


def test_symbols_and_bond_are_drawn():
    img = _pixels(_canvas())
    assert img[:, :, 3].max() == 255
    assert (img[:, :, 3] > 0).sum() > 1000  # symbols, bond and label all carry ink


def test_shadow_adds_ink_without_touching_the_symbols():
    plain, shady = _pixels(_canvas()), _pixels(_canvas(shadow = True))
    assert (shady[:, :, 3] > 0).sum() > (plain[:, :, 3] > 0).sum()
    ink = plain[:, :, 3] == 255
    assert np.array_equal(shady[:, :, :3][ink], plain[:, :, :3][ink])


def test_sticker_wraps_the_ink_in_a_white_border():
    cut = _pixels(_canvas(sticker = True))
    plain = _pixels(_canvas())
    grown = (cut[:, :, 3] > 250) & (plain[:, :, 3] == 0)
    assert grown.sum() > 500  # a band of new opaque pixels outside the original ink
    assert np.all(cut[:, :, :3][grown] > 230)  # and that band is white


def test_sticker_bridges_the_gap_between_the_symbols():
    col = int((-50.0 + 150.0) * 3.0)  # halfway along the bond, where only the 4-point line carries ink
    plain, cut = _pixels(_canvas())[:, col, 3], _pixels(_canvas(sticker = True))[:, col, 3]
    rows = np.nonzero(cut > 250)[0]
    assert np.all(np.diff(rows) == 1)  # one contiguous band: the cut follows the bond instead of leaving islands
    assert np.ptp(rows) > np.ptp(np.nonzero(plain > 250)[0]) + 2 * 0.18 * 50.0 * 3.0 * 0.9


def test_sticker_edge_adds_a_darker_keyline():
    white, keyed = _pixels(_canvas(sticker = True)), _pixels(_canvas(sticker = {'edge': 3.0}))
    dark = lambda img: ((img[:, :, 3] > 250) & (img[:, :, :3].max(axis = 2) < 80)).sum()
    assert (keyed[:, :, 3] > 250).sum() > (white[:, :, 3] > 250).sum() + 1000  # keyline sits outside the white band
    assert dark(keyed) > 2 * dark(white)


def test_sticker_params_scale_with_symbol_size_and_accept_overrides():
    ops = _canvas().ops
    spec = pdfwrite.sticker_params(ops, True)
    assert spec['width'] == pytest.approx(0.18 * 50.0)  # 50-unit symbols in the fixture
    assert spec['color'] == (1.0, 1.0, 1.0) and spec['edge'] == 0.0
    assert pdfwrite.sticker_params(ops, {'width': 3.0, 'color': (0.0, 0.0, 0.0)})['width'] == 3.0
    assert pdfwrite.sticker_params(ops, False) is None


def test_sticker_casts_the_shadow_unless_told_otherwise():
    shadow, sticker = _canvas(sticker = True)._effects()
    assert shadow is not None and sticker is not None
    assert _canvas(sticker = {'shadow': False})._effects()[0] is None
    lifted = _pixels(_canvas(sticker = True))
    flat = _pixels(_canvas(sticker = {'shadow': False}))
    assert (lifted[:, :, 3] > 0).sum() > (flat[:, :, 3] > 0).sum() + 1000  # the shadow falls outside the cut


def test_ink_mask_grows_by_exactly_the_requested_amount():
    c = _canvas()
    flip = (1.0, 0.0, 0.0, -1.0, 0.0, float(c.height))
    w, h = int(c.width), int(c.height)
    tight = raster._ink_mask(c.ops, w, h, flip, symbols_only = False)
    grown = raster._ink_mask(c.ops, w, h, flip, grow = 8.0, symbols_only = False)
    assert (grown > 0.5).sum() > (tight > 0.5).sum()
    row = int(tight.sum(axis = 1).argmax())
    span = lambda m: np.ptp(np.nonzero(m[row] > 0.5)[0])
    assert span(grown) == pytest.approx(span(tight) + 16.0, abs = 2.0)  # 8 points on either side


def test_svg_output_carries_the_cut_layer():
    plain, cut = _canvas().to_svg(), _canvas(sticker = True).to_svg()
    assert 'feDropShadow' not in plain and 'stroke="#FFFFFF"' not in plain
    assert cut.count('<g filter=') == 1  # one shadow for the whole cut, not one per piece
    assert 'stroke="#FFFFFF"' in cut and cut.index('#FFFFFF') < cut.index('#FCC326')  # cut sits behind the ink


def test_pdf_to_svg_bytes_passes_effects_through():
    assert 'feDropShadow' in pdf_to_svg_bytes(SNFG_SVG, sticker = True)
    assert 'feDropShadow' not in pdf_to_svg_bytes(SNFG_SVG)


def test_pdf_sticker_layer_stays_vector(tmp_path):
    plain, cut = tmp_path / "plain.pdf", tmp_path / "cut.pdf"
    convert_svg_to_pdf(SNFG_SVG, str(plain))
    convert_svg_to_pdf(SNFG_SVG, str(cut), sticker = True)
    streams = lambda p: b''.join(_inflate(part.split(b'\nendstream')[0]) for part in p.read_bytes().split(b'stream\n')[1:])
    plain_ops, cut_ops = streams(plain), streams(cut)
    assert b'1 1 1 rg' in cut_ops and b'1 J 1 j' in cut_ops  # white cut painted with round joins, as real path ops
    assert b'1 1 1 rg' not in plain_ops
    assert b'/Shadow' in cut.read_bytes()  # only the blurred shadow is rasterized