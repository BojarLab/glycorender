"""Tests for glycorender: the SVG subset GlycoDraw emits, and the PNG/PDF/SVG backends."""
import base64, importlib.metadata, math, os, re, shutil, struct, zlib
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import glycorender
from glycorender import pdfwrite, raster, svgdraw, svgin, ttf
from glycorender.pdfwrite import pdfmetrics
from glycorender.render import (_render_svg_to_pdf_canvas, convert_svg_to_pdf, convert_svg_to_png, font_to_use,
                                parse_color, parse_path, pdf_to_svg_bytes, simple_svg_to_pdf, simple_svg_to_png)

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


def test_pdf_subset_font_is_tagged_by_its_glyph_set(tmp_path):
    tags = []
    for i, label in enumerate(('b 3', 'b 3', '3 b', 'a 6')):
        out = tmp_path / ("%d.pdf" % i)
        convert_svg_to_pdf(SNFG_SVG.replace('b 3', label), str(out))
        names = re.findall(rb'/(?:BaseFont|FontName) /(\S+)', out.read_bytes())
        # Type0 font, CIDFont and FontDescriptor all carry the same TAG+Font name (ISO 32000-1, 9.6.4)
        assert len(names) == 3 and len(set(names)) == 1 and re.fullmatch(rb'[A-Z]{6}\+\S+', names[0])
        tags.append(names[0])
    # Reproducible, the same for the same glyphs in any order, and different for a different subset
    assert tags[0] == tags[1] == tags[2] != tags[3]


def test_gpos_kerning_takes_the_first_subtable_and_adds_up_lookups():
    font = ttf.TTF(os.path.join(os.path.dirname(ttf.__file__), 'fonts', 'Comfortaa-Regular.ttf'))
    # Comfortaa lists 'Q.' as an exception (-60) ahead of the class kerning, which would give it +10
    assert font.kern(font.gid('Q'), font.gid('.')) == -60
    # A GPOS whose kern feature has two lookups for glyphs 1 and 2: -30 then +5 in the first, -20 in the second
    pair_pos = lambda adj: struct.pack('>6H2Hh3H', 1, 18, 4, 0, 1, 12, 1, 2, adj, 1, 1, 1)
    lookup = lambda *adjs: (struct.pack('>3H', 2, 0, len(adjs)) + struct.pack(
        '>%dH' % len(adjs), *(6 + 2 * len(adjs) + 24 * j for j in range(len(adjs)))) + b''.join(map(pair_pos, adjs)))
    first, second = lookup(-30, 5), lookup(-20)
    gpos = (struct.pack('>5H', 1, 0, 0, 10, 26) + struct.pack('>H4sH', 1, b'kern', 8) + struct.pack('>4H', 0, 2, 0, 1)
            + struct.pack('>3H', 2, 6, 6 + len(first)) + first + second)
    assert ttf._read_kerning(SimpleNamespace(data = gpos, tables = {'GPOS': (0, len(gpos))})) == {(1, 2): -50}


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


def test_box_blur_is_three_moving_averages_per_axis():
    a = np.random.RandomState(0).rand(23, 31)
    for radius in (1, 2, 6):
        ref, k = a, 2 * radius + 1
        for _ in range(3):
            for axis in (0, 1):
                ref = np.apply_along_axis(lambda r: np.convolve(r, np.ones(k) / k, 'same'), axis, ref)
        assert np.allclose(raster._box_blur(a, radius), ref, rtol = 0, atol = 1e-6)


def test_round_joins_fill_the_whole_stroke_outline():
    # Left and right turns, sharp and shallow: with round caps and joins the stroke is exactly the distance field
    pts, hw = [(10.0, 10.0), (40.0, 30.0), (15.0, 45.0), (50.0, 55.0), (52.0, 20.0), (58.0, 19.0)], 6.0
    a = raster.coverage(raster._edges(raster.stroke_polys([pts], 2 * hw, 1, 1), True), 0, 0, 72, 72)
    px, py = np.meshgrid(np.arange(72) + 0.5, np.arange(72) + 0.5)
    dist = np.full(px.shape, np.inf)
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        t = np.clip(((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2), 0, 1)
        dist = np.minimum(dist, np.hypot(px - ax - t * (bx - ax), py - ay - t * (by - ay)))
    assert np.all(a[dist < hw - 0.75] > 0.99) and np.all(a[dist > hw + 0.75] < 0.01)


def test_convex_outlines_cover_what_the_stroke_pieces_cover():
    t = np.linspace(0, 2 * np.pi, 257)
    circle = list(zip((60.3 + 40 * np.cos(t)).tolist(), (60.7 + 40 * np.sin(t)).tolist()))
    circle[-1] = circle[0]
    square = [(20.0, 20.0), (100.0, 20.0), (100.0, 100.0), (20.0, 100.0), (20.0, 20.0)]
    spike = [(60.0, 5.0), (64.0, 110.0), (56.0, 110.0), (60.0, 5.0)]  # an apex past the miter limit is beveled
    seg = [(20.2, 30.1), (95.3, 80.7)]
    def both(sub, hw, cap, join, filled = False):
        a = raster._outline([sub], hw, cap, join, filled)
        b = raster._edges(raster.stroke_polys([sub], 2 * hw, cap, join) + ([sub] if filled else []), True)
        return raster.coverage(a, 0, 0, 130, 130, simple = True), raster.coverage(b, 0, 0, 130, 130)
    # Straight sides: the same region, exactly
    for sub, hw, cap, join, filled in [(square, 4, 0, 0, False), (square[::-1], 4, 0, 1, False),
                                       (square, 9, 0, 1, True),
                                       (spike, 2, 0, 0, False), (spike, 2, 0, 2, False), (seg, 4, 0, 0, False),
                                       (seg, 4, 2, 0, False)]:
        a, b = both(sub, hw, cap, join, filled)
        assert np.abs(a - b).max() < 1e-9
    # Curves and round ends: the same ink, minus what the overlapping pieces double-counted along their edges
    for sub, cap, join in [(circle, 0, 1), (circle, 0, 0), (seg, 1, 1), (seg + seg[:1], 1, 1), (seg + seg[:1], 1, 0)]:
        a, b = both(sub, 4, cap, join)
        assert a.sum() == pytest.approx(b.sum(), rel = 0.05) and a.sum() <= b.sum() + 0.5
    star = [(60 + (40 if i % 2 else 15) * np.cos(i * np.pi / 5), 60 + (40 if i % 2 else 15) * np.sin(i * np.pi / 5))
            for i in range(11)]
    small = [(x / 8 + 50, y / 8 + 50) for x, y in circle]
    back = [(60.0, 10.0), (60.0, 40.0), (60.0, 90.0), (60.0, 10.0)]
    # Out and back on a diagonal, collinear only up to rounding, so that it turns by 4e-16 less than pi
    diag = [(13.025 + (97.3 - 13.025) * s, 21.7 + (113.9 - 21.7) * s) for s in np.linspace(0, 1, 33).tolist()]
    diag.append(diag[0])
    with np.errstate(divide = 'raise', invalid = 'raise'):
        for subs, hw in [([star], 2), ([square, square], 2), ([square[:3]], 2), ([small], 6), ([back], 1), ([diag], 1)]:
            # concave, two subpaths, open, eroded away, out and back along one line: stroke_polys
            assert raster._outline(subs, hw, 1, 1) is None
    assert raster._convex(star) is None and raster._convex(back) is None and raster._convex(square[::-1]) is not None
    a = raster.coverage(raster._edges(raster.stroke_polys([back], 2, 0, 1), True), 0, 0, 130, 130)
    assert np.isfinite(a).all() and a[20:80, 59:61].min() > 0.4


def test_coverage_in_row_chunks_with_both_fill_rules():
    # 20000 pixels wide, so the sub-scanlines come in several chunks; an inner square, the same way round, is a hole
    # only under even-odd
    outer = [(10.0, 5.0), (19990.0, 5.0), (19990.0, 55.0), (10.0, 55.0)]
    inner = [(100.0, 20.0), (200.0, 20.0), (200.0, 40.0), (100.0, 40.0)]
    edges = raster._edges([outer, inner], True)
    assert raster.coverage(edges, 0, 0, 20000, 60).sum() == pytest.approx(19980 * 50)
    assert raster.coverage(edges, 0, 0, 20000, 60, even_odd = True).sum() == pytest.approx(19980 * 50 - 2000)
    assert raster.coverage(raster._edges([outer], True), 0, 0, 20000, 60, simple = True).sum() == pytest.approx(
        19980 * 50)


def test_gradient_concave_and_dashed_shapes_reach_the_png():
    svg = SNFG_SVG.replace('<defs>', '<defs><radialGradient id="g" cx="0.5" cy="0.5" r="0.5"><stop offset="0" '
                           'stop-color="#FF0000" /><stop offset="1" stop-color="#0000FF" /></radialGradient>').replace(
        '</g>', '<circle cx="20" cy="30" r="15" fill="url(#g)" /><path d="M-140,-40 L-120,-50 L-130,-30 L-110,-35 Z" '
        'fill="#FCC326" stroke="#000000" stroke-width="2.0" /><path d="M-140,45 L40,45" stroke="#000000" '
        'stroke-width="2.0" stroke-dasharray="10,10" fill="none" /></g>')
    c = _render_svg_to_pdf_canvas(svg, None, alt_text_info = None)
    img = raster.render(c.ops, c.width, c.height, 2.0, 2.0, (1.0, 1.0, 1.0))
    assert img[170, 340, 0] > 240 and img[170, 340, 2] < 15  # the gradient's center
    assert img[200, 22:280].min() < 100 and (img[200, 22:280].min(axis = 1) > 200).sum() > 60  # dashes and gaps
    sticker = raster._ink_mask(c.ops, 400, 220, (2.0, 0.0, 0.0, -2.0, 0.0, 220.0), grow = 6.0, symbols_only = False)
    assert sticker[16:24, 30].max() > 0.99  # the concave quadrilateral, grown


def test_svg_output_carries_the_cut_layer():
    plain, cut = _canvas().to_svg(), _canvas(sticker = True).to_svg()
    assert 'feDropShadow' not in plain and 'stroke="#FFFFFF"' not in plain
    assert cut.count('<g filter=') == 1  # one shadow for the whole cut, not one per piece
    assert 'stroke="#FFFFFF"' in cut and cut.index('#FFFFFF') < cut.index('#FCC326')  # cut sits behind the ink


def test_svg_output_writes_shared_transform_and_cut_paint_once():
    plain, cut = _canvas().to_svg(), _canvas(sticker = True).to_svg()
    edged = _canvas(sticker = {'edge': 2.0}).to_svg()
    assert cut.split('\n')[1] == edged.split('\n')[1] == plain.split('\n')[1]  # the <svg> element: same size either way
    # The drawing's group applies the transform the shapes share; only the label, turned upright, keeps one (cut and ink).
    # So the cut's shadow is cast in the shapes' own unflipped space and falls down the page, as in the PNG and PDF
    assert cut.count('transform="matrix(') == 3 and '<g transform="matrix(1 0 0 1 ' in cut
    # A cut layer's group carries its paint, so its pieces carry only geometry and width
    assert cut.count('stroke="#FFFFFF"') == 1 and cut.count('fill="none"') == 2
    assert edged.count('<g fill=') == 2 and edged.count('<g filter=') == 1  # keyline and cut share one shadow


def test_svg_image_is_pixelated_and_has_no_vote_on_the_shared_transform():
    c = pdfwrite.Canvas(BytesIO(), pagesize = (100, 100))
    op = dict(c.state)
    op.update(kind = 'image', w = 1, h = 1, rgb = bytes(3), alpha = None, ctm = (50.0, 0.0, 0.0, 50.0, 0.0, 0.0))
    c.ops.append(op)
    assert 'image-rendering:pixelated' in c.to_svg() and '<g transform="matrix(1 0 0 -1 0 100)">' in c.to_svg()
    p = c.beginPath()
    p.moveTo(1.0, 1.0)
    p.lineTo(9.0, 1.0)
    p.lineTo(9.0, 9.0)
    p.close()
    c.drawPath(p, fill = 1, stroke = 0)
    svg = c.to_svg()
    assert '<g transform="matrix(1 0 0 -1 0 100)">' in svg and '<path d="M1 1L9 1L9 9Z" fill=' in svg


def test_svg_clip_rect_keeps_its_own_transform(tmp_path):
    simple_svg_to_png('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><defs><clipPath id="k">'
                      '<rect width="5" height="20" transform="matrix(1 0 0 1 10 0)"/></clipPath></defs>'
                      '<rect width="20" height="20" fill="#FF0000" clip-path="url(#k)"/></svg>', tmp_path / "clip.png")
    c = pdfwrite.Canvas(None, pagesize = (1, 1))
    c.drawImage((tmp_path / "clip.png").read_bytes())
    h, w = c.ops[0]['h'], c.ops[0]['w']
    alpha = np.frombuffer(c.ops[0]['alpha'], np.uint8).reshape(h, w)
    assert alpha[h // 2, w * 12 // 20] == 255 and alpha[h // 2, w * 2 // 20] == 0


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


def test_png_records_its_resolution():
    png = convert_svg_to_png(SNFG_SVG, None, scale = 300 / 72, return_bytes = True)
    i = png.index(b'pHYs')
    assert round(int.from_bytes(png[i + 4:i + 8], 'big') * 0.0254) == 300 and png[i + 12] == 1  # pixels per metre


def test_draw_image_undoes_every_png_filter():
    rng = np.random.RandomState(0)
    chunk = lambda tag, body: struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body))
    png = lambda head, raw: (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', *head))
                             + chunk(b'IDAT', raw[:9]) + chunk(b'IDAT', raw[9:]) + chunk(b'IEND', b''))
    c = pdfwrite.Canvas(None, pagesize = (10, 10))
    for ctype, bpp in ((0, 1), (2, 3), (4, 2), (6, 4)):
        # Five rows of noise, then ten of few levels, where Paeth's tie-breaking order decides
        src = np.vstack([rng.randint(0, 256, (5, 3 * bpp)), rng.randint(0, 4, (10, 3 * bpp)) * 85])
        rows, prev = [], np.zeros(3 * bpp, dtype = int)
        # Row r uses filter r (None, Sub, Up, Average, Paeth, then Paeth), predicted from its unfiltered neighbors
        for r, row in enumerate(src):
            left = np.concatenate([np.zeros(bpp, dtype = int), row[:-bpp]])
            corner = np.concatenate([np.zeros(bpp, dtype = int), prev[:-bpp]])
            p = left + prev - corner
            paeth = np.where((abs(p - left) <= abs(p - prev)) & (abs(p - left) <= abs(p - corner)), left,
                             np.where(abs(p - prev) <= abs(p - corner), prev, corner))
            pred = (0, left, prev, (left + prev) // 2, paeth)[min(r, 4)]
            rows.append(bytes([min(r, 4)]) + ((row - pred) % 256).astype(np.uint8).tobytes())
            prev = row
        c.drawImage(png((3, 15, 8, ctype, 0, 0, 0), zlib.compress(b''.join(rows))))
        op, px = c.ops[-1], src.reshape(15, 3, bpp).astype(np.uint8)
        assert (op['kind'], op['w'], op['h']) == ('image', 3, 15)
        assert op['rgb'] == (px[:, :, :3] if bpp > 2 else px[:, :, :1].repeat(3, axis = 2)).tobytes()
        assert op['alpha'] == (px[:, :, -1].tobytes() if bpp in (2, 4) else None)
    # Paletted, interlaced and 16-bit PNGs, and anything that is no PNG, are skipped rather than drawn wrong
    for head in ((3, 5, 8, 3, 0, 0, 0), (3, 5, 8, 2, 0, 0, 1), (3, 5, 16, 2, 0, 0, 0)):
        c.drawImage(png(head, zlib.compress(bytes(100))))
    c.drawImage(b'GIF89a')
    assert len(c.ops) == 4


def test_embedded_png_images_reach_png_and_pdf(tmp_path):
    chunk = lambda tag, body: struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body))
    png = lambda px: base64.b64encode(
        b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 6, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(b''.join(b'\x00' + r.tobytes() for r in px))) + chunk(b'IEND', b'')).decode()
    rgba = np.array([[[230, 30, 40, 255], [20, 160, 60, 255]], [[30, 60, 200, 255], [250, 200, 0, 128]]], np.uint8)
    opaque = rgba.copy()
    opaque[:, :, 3] = 255
    # matplotlib wraps its base64 across lines; the second image is clipped away entirely, the JPEG is skipped
    b64 = png(rgba)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="40" height="40">'
           '<defs><clipPath id="k"><rect x="0" y="0" width="5" height="5"/></clipPath></defs>'
           '<image x="10" y="10" width="20" height="20" xlink:href="data:image/png;base64,%s\n%s"/>'
           '<image x="10" y="10" width="20" height="20" clip-path="url(#k)" xlink:href="data:image/png;base64,%s"/>'
           '<image x="0" y="0" width="40" height="40" xlink:href="data:image/jpeg;base64,/9j/"/></svg>'
           % (b64[:20], b64[20:], png(opaque)))
    simple_svg_to_png(svg, tmp_path / "fig.png")
    c = pdfwrite.Canvas(None, pagesize = (1, 1))
    c.drawImage((tmp_path / "fig.png").read_bytes())
    out = np.frombuffer(c.ops[0]['rgb'], np.uint8).reshape(c.ops[0]['h'], c.ops[0]['w'], 3)
    alpha = np.frombuffer(c.ops[0]['alpha'], np.uint8).reshape(out.shape[:2])
    # Page units 15 and 25 are the centers of the four image pixels, at 300 dpi; the first image row is its top
    at = lambda u: int(u * 300 / 72)
    for (i, j), (y, x) in zip(np.ndindex(2, 2), ((15, 15), (15, 25), (25, 15), (25, 25))):
        assert out[at(y), at(x)] == pytest.approx(rgba[i, j, :3], abs = 1)
        assert alpha[at(y), at(x)] == pytest.approx(rgba[i, j, 3], abs = 1)
    assert alpha[at(5), at(5)] == 0
    simple_svg_to_pdf(svg, tmp_path / "fig.pdf")
    data = (tmp_path / "fig.pdf").read_bytes()
    streams = [_inflate(part.split(b'\nendstream')[0]) for part in data.split(b'stream\n')[1:]]
    assert b'/Im0 Do' in b''.join(streams) and b'/Im1 Do' in b''.join(streams)
    # Both images become XObjects, but only the translucent one needs a soft mask
    assert data.count(b'/Subtype /Image') == 3 and data.count(b'/SMask') == 1
    # Their rows come filtered as in a PNG, by None or Up
    assert data.count(b'/Predictor 15') == 3
    def rows(stream, stride):
        out, prev = [], np.zeros(stride, np.uint8)
        for r in range(len(stream) // (stride + 1)):
            line = np.frombuffer(stream[r * (stride + 1) + 1:(r + 1) * (stride + 1)], np.uint8)
            prev = line + prev if stream[r * (stride + 1)] == 2 else line.copy()
            out.append(prev.tobytes())
        return b''.join(out)
    assert rgba[:, :, :3].tobytes() in [rows(s, 6) for s in streams if len(s) == 14]
    assert rgba[:, :, 3].tobytes() in [rows(s, 2) for s in streams if len(s) == 6]


def test_labels_never_read_right_to_left():
    for turn in (0, 45, 90, 135, 180, 225, 270, 315):
        c = _render_svg_to_pdf_canvas(SNFG_SVG.replace('rotate(0 ', 'rotate(%d ' % turn), None)
        assert all(op['ctm'][0] > -1e-6 for op in c.ops if op['kind'] == 'text')


def test_ring_label_at_the_origin_is_drawn():
    carrier = '<path d="M-50,0 L50,0" stroke-width="0" id="d1" />\n</defs>'
    label = ('<text font-size="15.0" fill="#000000" text-anchor="middle"><textPath xlink:href="#d1" startOffset="50%">'
             '<tspan dy="0.5em">Lf</tspan></textPath></text>\n</g>')
    svg = SNFG_SVG.replace('</defs>', carrier).replace('</g>', label)
    texts = [op for op in _render_svg_to_pdf_canvas(svg, None).ops if op['kind'] == 'text']
    assert [op['text'] for op in texts] == ['b3', 'L', 'f']
    assert texts[2]['ctm'][2] != 0  # the furanose f is skewed into a faked italic


def test_svgdraw_writes_what_drawsvg_wrote():
    d = svgdraw.Drawing(200, 110, origin=(-150.0, -55.0))
    g = svgdraw.Group(transform='rotate(0 -50.0 0.0)')
    bond = svgdraw.Path(stroke_width=4.0, stroke='#000000').M(-100, 0).L(0, 0)
    g.append(bond)
    g.append(svgdraw.Text('b 3', 20.0, path=bond, text_anchor='middle', fill='#000000', valign='middle',
                          line_offset=-0.5))
    g.append(svgdraw.Rectangle(-25.0, -25.0, 50, 50, fill='#FCC326', stroke_width=2.0, stroke='#000000'))
    g.append(svgdraw.Circle(-100, 0, 25.0, fill='#0090BC', stroke_width=2.0, stroke='#000000'))
    d.append(g)
    assert d.as_svg() == SNFG_SVG


def test_svgdraw_defs_duplicates_and_text_forms():
    # Expected output is what drawsvg 2.4 writes for the same calls
    d = svgdraw.Drawing(10, 10, id_prefix='g_')
    carrier = svgdraw.Path().M(0, 0).L(1, 1)
    glow = svgdraw.RadialGradient(1, 2, 3)
    glow.add_stop(0, 'red', opacity=0.5)
    twice, own = svgdraw.Circle(0, 0, 1), svgdraw.Circle(0, 0, 2, id='own')
    shape = svgdraw.Lines(0, 0, 1, 1, 2, 0, close=True).C(1, 2, 3, 4, 5, 6)
    d.append(svgdraw.Group([svgdraw.Text('A&B', 5, path=carrier, text_anchor='end', line_offset=-3.15),
                            svgdraw.Text('f', 5, path=carrier, center=True), svgdraw.Text('', 5, path=carrier),
                            svgdraw.Text('a\nb', 5, 1, 2), svgdraw.Text('c', 5, 1, 2, center=True),
                            svgdraw.Circle(1, 2, 3, fill=glow), twice, twice, own, own, shape]))
    assert d.as_svg() == """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="10" height="10" viewBox="0 0 10 10">
<defs>
<path d="M0,0 L1,1" id="g_0" />
<radialGradient cx="1" cy="2" r="3" gradientUnits="userSpaceOnUse" id="g_1">
<stop offset="0" stop-color="red" stop-opacity="0.5" />
</radialGradient>
</defs>
<g>
<text font-size="5" text-anchor="end"><textPath xlink:href="#g_0" startOffset="100%">
<tspan dy="-3.15em" dx="-1">A&amp;B</tspan>
</textPath></text>
<text font-size="5" text-anchor="middle"><textPath xlink:href="#g_0" startOffset="50%">
<tspan dy="0.5em">f</tspan>
</textPath></text>
<text font-size="5"><textPath xlink:href="#g_0"></textPath></text>
<text x="1" y="2" font-size="5"><tspan x="1" dy="0em">a</tspan><tspan x="1" dy="1em">b</tspan></text>
<text x="1" y="2" font-size="5" text-anchor="middle" dominant-baseline="central">c</text>
<circle cx="1" cy="2" r="3" fill="url(#g_1)" />
<circle cx="0" cy="0" r="1" id="g_2" />
<use xlink:href="#g_2" />
<circle cx="0" cy="0" r="2" id="own" />
<use xlink:href="#own" />
<path d="M0,0 L1,1 L2,0 Z C1,2,3,4,5,6" />
</g>
</svg>"""
    assert d.all_elements() == d.elements and len(_render_svg_to_pdf_canvas(d.as_svg(), None).ops) > 0


def _image_op(w, h, ctm, alpha = None, fill_alpha = 1.0):
    # A page's display list holding one image, with a color per pixel from its position
    rgb = np.dstack([np.arange(w * h).reshape(h, w) % 251, np.arange(w * h).reshape(h, w) % 241,
                     np.arange(w * h).reshape(h, w) % 239]).astype(np.uint8)
    c = pdfwrite.Canvas(BytesIO(), pagesize = (100, 100))
    op = dict(c.state)
    op.update(kind = 'image', w = w, h = h, rgb = rgb.tobytes(), alpha = None if alpha is None else alpha.tobytes(),
              ctm = ctm, fill_alpha = fill_alpha)
    return op, rgb


def _rotated_route(op, scale):
    # A rotation too small to move a pixel takes the general route, which the upright one has to agree with
    tilted = dict(op)
    a, b, c, d, e, f = op['ctm']
    tilted['ctm'] = (a, 1e-11, 1e-11, d, e, f)
    return raster.render([tilted], 100, 100, scale, scale, (1.0, 1.0, 1.0))


@pytest.mark.parametrize('ctm, scale', [((50.0, 0.0, 0.0, 50.0, 10.0, 20.0), 1.0),  # on the output's grid
                                        ((37.31, 0.0, 0.0, 61.73, 10.1, 20.4), 2.0),  # stretched
                                        ((-40.0, 0.0, 0.0, 40.0, 60.0, 20.0), 1.0),  # mirrored
                                        ((40.0, 0.0, 0.0, -40.0, 20.0, 70.0), 1.0),
                                        ((300.0, 0.0, 0.0, 300.0, -100.0, -100.0), 1.0)])  # past the page
@pytest.mark.parametrize('kind', ['opaque', 'dense', 'sparse', 'clear', 'half'])
def test_upright_image_route_matches_general_route(ctm, scale, kind):
    w, h = 50, 50
    rng = np.random.default_rng(0)
    alpha = {'opaque': None, 'half': None, 'dense': rng.integers(0, 256, (h, w)).astype(np.uint8),
             'sparse': (rng.random((h, w)) < 0.05) * rng.integers(1, 256, (h, w)).astype(np.uint8),
             'clear': np.zeros((h, w), np.uint8)}[kind]
    op, _ = _image_op(w, h, ctm, alpha, 0.5 if kind == 'half' else 1.0)
    fast = raster.render([op], 100, 100, scale, scale, (1.0, 1.0, 1.0))
    assert np.abs(fast.astype(int) - _rotated_route(op, scale).astype(int)).max() <= 1


def test_image_on_grid_copies_its_pixels():
    op, rgb = _image_op(30, 20, (30.0, 0.0, 0.0, 20.0, 5.0, 7.0))
    out = raster.render([op], 100, 100, 1.0, 1.0, (1.0, 1.0, 1.0))
    assert (out[100 - 27:100 - 7, 5:35] == rgb).all() and (out[:100 - 27] == 255).all()


def test_image_entirely_off_the_page_or_transparent():
    op, _ = _image_op(10, 10, (10.0, 0.0, 0.0, 10.0, 200.0, 200.0))
    assert (raster.render([op], 100, 100, 1.0, 1.0, (1.0, 1.0, 1.0)) == 255).all()
    op, _ = _image_op(10, 10, (10.0, 0.0, 0.0, 10.0, 5.0, 5.0), fill_alpha = 0.0)
    assert (raster.render([op], 100, 100, 1.0, 1.0, (1.0, 1.0, 1.0)) == 255).all()


@pytest.mark.parametrize('coords', [(0.0, 10.0, 0.0, 90.0), (10.0, 0.0, 90.0, 0.0), (10.0, 10.0, 90.0, 90.0),
                                    (50.0, 50.0, 50.0, 50.0)])
def test_linear_gradient_along_an_axis(coords):
    c = pdfwrite.Canvas(BytesIO(), pagesize = (100, 100))
    c.setFillGradient('linear', coords, [(0.0, (1.0, 0.0, 0.0, 1.0)), (1.0, (0.0, 0.0, 1.0, 1.0))])
    c.rect(10, 10, 80, 80, fill = 1, stroke = 0)
    upright = c.ops[0]
    tilted = dict(upright, ctm = (1.0, 1e-11, 1e-11, 1.0, 0.0, 0.0))
    fast = raster.render([upright], 100, 100, 2.0, 2.0, (1.0, 1.0, 1.0))
    assert np.abs(fast.astype(int) - raster.render([tilted], 100, 100, 2.0, 2.0, (1.0, 1.0, 1.0)).astype(int)).max() <= 1


def test_large_path_coverage_matches_by_band():
    # Enough rows to take several bands, computed in threads; each band's result must not depend on the others
    t = np.linspace(0.0, 40.0, 400)
    subs = [list(zip((t * 40.0).tolist(), (1000.0 + 900.0 * np.sin(t)).tolist()))]
    edges = raster._edges(raster.stroke_polys(subs, 8.0, 1, 1), True)
    whole = raster.coverage(edges, 0, 0, 1700, 2000)
    assert whole.shape == (2000, 1700)
    tops = [raster.coverage(edges, 0, y, 1700, 100) for y in (0, 1000, 1900)]
    for y, part in zip((0, 1000, 1900), tops):
        assert np.array_equal(whole[y:y + 100], part)


def _star(n, r):
    return [[(1000 + r * math.cos(k * n * math.pi / 5), 1000 + r * math.sin(k * n * math.pi / 5)) for k in range(6)]]


@pytest.mark.parametrize('shape', ['stroke', 'clipped', 'ring', 'star_nonzero', 'star_even_odd', 'two_rings', 'disc'])
def test_sparse_coverage_equals_the_whole_array(shape):
    # Thin ink over a large box comes back as its covered pixels, which must be exactly the array's nonzero ones
    t = np.linspace(0.0, 40.0, 400)
    line = [list(zip((50.0 + t * 45.0).tolist(), (1000.0 + 900.0 * np.sin(t)).tolist()))]
    ring = raster._arc(1000, 1000, 900)
    simple, even_odd, box = False, False, (0, 0, 1900, 2000)
    if shape in ('stroke', 'clipped'):
        edges = raster._edges(raster.stroke_polys(line, 6.0, 1, 1), True)
        # A box that ends before the ink does clips its crossings to its right side
        box = (0, 0, 1900, 2000) if shape == 'stroke' else (37, 13, 1500, 1900)
    elif shape == 'ring':
        edges, simple = raster._outline([ring + [ring[0]]], 4.0, 1, 1), True
    elif shape == 'disc':
        edges, simple = raster._edges([ring], True), True
    elif shape == 'two_rings':
        edges = raster._edges(raster.stroke_polys([ring + [ring[0]], raster._arc(1000, 1000, 500)[::-1]], 9.0, 0, 0),
                              True)
    else:
        edges, even_odd = raster._edges(_star(4, 950), True), shape == 'star_even_odd'
    whole = raster.coverage(edges, *box, even_odd, simple)
    found = raster.coverage(edges, *box, even_odd, simple, True)
    if shape == 'disc':
        # Mostly covered, so the whole array comes back
        assert isinstance(found, np.ndarray) and np.array_equal(found, whole)
        return
    ys, xs, v = found
    assert len(ys) == np.count_nonzero(whole) and len(set((ys * box[2] + xs).tolist())) == len(ys)
    assert np.array_equal(whole[ys, xs], v)


def test_sparse_coverage_paints_the_same_pixels():
    t = np.linspace(0.0, 40.0, 400)
    c = pdfwrite.Canvas(BytesIO(), pagesize = (500, 500))
    c.setStrokeColorRGB(0.1, 0.5, 0.6, alpha = 0.3)
    c.setLineWidth(3.0)
    p = c.beginPath()
    p.moveTo(10, 250)
    for x, y in zip(10 + t * 12, 250 + 220 * np.sin(t)):
        p.lineTo(x, y)
    c.drawPath(p, fill = 0, stroke = 1)
    sparse = raster.render(c.ops, 500, 500, 4.0, 4.0, (1.0, 1.0, 1.0))
    with patch.object(raster, 'SPARSE_MIN', 1 << 40):
        assert np.array_equal(sparse, raster.render(c.ops, 500, 500, 4.0, 4.0, (1.0, 1.0, 1.0)))


def test_a_layer_goes_under_the_drawing():
    c = pdfwrite.Canvas(BytesIO(), pagesize = (10, 10))
    c.setFillColorRGB(1.0, 0.0, 0.0)
    c.rect(0, 0, 5, 10, fill = 1, stroke = 0)
    # Half-transparent blue over the whole page, as premultiplied planes, from the third row and second column on
    planes = np.zeros((4, 10, 10), np.float32)
    planes[2], planes[3] = 0.5, 0.5
    got = raster.render(c.ops, 10, 10, 1.0, 1.0, (1.0, 1.0, 1.0), under = lambda: (2, 1, planes))
    assert tuple(got[5, 2]) == (255, 0, 0) and tuple(got[0, 7]) == (255, 255, 255)
    assert tuple(got[5, 7]) == (128, 128, 255) and tuple(got[1, 7]) == (255, 255, 255)
    # A layer reaching past the page is cut to it, and nothing at all is the same as no layer
    assert raster.render(c.ops, 10, 10, 1.0, 1.0, None, under = (-3, 8, planes)).shape == (10, 10, 4)
    assert np.array_equal(raster.render(c.ops, 10, 10, 1.0, 1.0, None, under = lambda: None),
                          raster.render(c.ops, 10, 10, 1.0, 1.0, None))


@pytest.mark.parametrize('size', [(40, 30), (1400, 900)])
def test_png_roundtrip_small_and_threaded(size):
    # A large image goes through zlib in pieces, which has to stay one valid stream
    w, h = size
    arr = (np.arange(w * h * 3).reshape(h, w, 3) % 251).astype(np.uint8)
    png = raster.encode_png(arr)
    pos, idat = 8, []
    while pos < len(png):
        n, tag = struct.unpack('>I4s', png[pos:pos + 8])
        if tag == b'IDAT':
            idat.append(png[pos + 8:pos + 8 + n])
        pos += 12 + n
    raw = zlib.decompress(b''.join(idat))
    assert len(raw) == h * (w * 3 + 1) and set(raw[::w * 3 + 1]) <= {0, 2}
    c = pdfwrite.Canvas(None, pagesize = (1, 1))
    c.drawImage(png)
    assert c.ops[0]['rgb'] == arr.tobytes()


def test_png_filters_each_band_by_what_packs_smaller():
    # A soft blur packs far smaller as differences to the row above, noise as it is
    y, x = np.mgrid[:600, :800]
    smooth = (128 + 100 * np.sin(x / 50.0) * np.cos(y / 70.0))[..., None].repeat(3, axis = 2).astype(np.uint8)
    noise = np.random.default_rng(0).integers(0, 256, (600, 800, 3), dtype = np.uint8)
    for arr, kind in ((smooth, 2), (noise, 0)):
        png = raster.encode_png(arr)
        raw = zlib.decompress(png[png.index(b'IDAT') + 4:png.index(b'IEND') - 8])
        assert set(raw[::800 * 3 + 1]) == {kind}
    assert len(raster.encode_png(smooth)) < 0.95 * len(zlib.compress(smooth.tobytes(), 6))


def _pdf_streams(data):
    return [zlib.decompress(m.group(1)) for m in re.finditer(rb'stream\n(.*?)\nendstream', data, re.S)
            if m.group(1)[:1] == b'\x78']


@pytest.mark.parametrize('fill, stroke, even_odd, paint', [(1, 0, False, b'f'), (1, 1, False, b'B'), (0, 1, False, b'S'),
                                                           (1, 1, True, b'B*'), (1, 0, True, b'f*')])
def test_markers_are_one_form_in_pdf(fill, stroke, even_odd, paint):
    target = BytesIO()
    c = pdfwrite.Canvas(target, pagesize = (100, 100))
    c.state['even_odd'] = even_odd
    c.setFillColorRGB(0.5, 0.25, 0.75, alpha = 0.5)
    c.setStrokeColorRGB(0.1, 0.2, 0.3)
    c.setLineWidth(2.0)
    p = c.beginPath()
    p.moveTo(-1.0, -1.0)
    p.lineTo(1.0, -1.0)
    p.lineTo(0.0, 1.0)
    p.close()
    c.drawMarkers(p, [10.0, 20.5], [30.0, 40.25], fill = fill, stroke = stroke)
    # A second batch of the same marker shares the form, an empty one draws nothing
    c.drawMarkers(p, [50.0], [60.0], fill = fill, stroke = stroke)
    c.drawMarkers(c.beginPath(), [50.0], [60.0])
    c.save()
    data = target.getvalue()
    assert data.count(b'/Subtype /Form') == 1
    streams = _pdf_streams(data)
    page = next(s for s in streams if b'Do Q' in s)
    assert page.count(b'/M0 Do') == 3 and b'q 1 0 0 1 20.5 40.25 cm /M0 Do Q' in page
    # The paint operator belongs to the form alone: on the page there is no path left for it to paint
    assert b'Do Q\n' + paint + b'\n' not in page
    form = next(s for s in streams if s.endswith(b'\n' + paint))
    assert form.startswith(b'-1 -1 m')
    margin = 10 if stroke else 0
    assert ('/BBox [%d %d %d %d]' % (-1 - margin, -1 - margin, 1 + margin, 1 + margin)).encode() in data


def test_shape_too_thin_to_cross_a_sub_scanline_draws_nothing():
    c = pdfwrite.Canvas(BytesIO(), pagesize = (10, 10))
    c.setFillColorRGB(0.0, 0.0, 0.0)
    p = c.beginPath()
    p.moveTo(1.0, 5.01)
    p.lineTo(9.0, 5.01)
    p.lineTo(9.0, 5.02)
    p.lineTo(1.0, 5.02)
    p.close()
    c.drawPath(p, fill = 1, stroke = 0)
    assert not raster.render(c.ops, 10, 10)[:, :, 3].any()


def test_image_alpha_only_gets_a_mask_when_it_has_transparency():
    for alpha, masks in ((bytes([255]) * 4, 0), (bytes([255, 255, 255, 3]), 1), (None, 0)):
        target = BytesIO()
        c = pdfwrite.Canvas(target, pagesize = (10, 10))
        op = dict(c.state)
        op.update(kind = 'image', w = 2, h = 2, rgb = bytes(12), alpha = alpha, ctm = (10.0, 0.0, 0.0, 10.0, 0.0, 0.0))
        c.ops.append(op)
        c.save()
        assert target.getvalue().count(b'/SMask') == masks


def test_very_large_page_compresses_at_a_lower_level():
    # Both levels give a valid stream; only the small page keeps level 9, so that GlycoDraw's PDFs stay byte-identical
    target = BytesIO()
    c = pdfwrite.Canvas(target, pagesize = (100, 100))
    p = c.beginPath()
    p.moveTo(0.0, 0.0)
    for i in range(120000):
        p.lineTo(i % 97 + 0.25, i % 89 + 0.5)
    c.drawPath(p, fill = 0, stroke = 1)
    c.save()
    assert len(_pdf_streams(target.getvalue())[0]) > 1 << 20

def test_a_page_of_millions_of_pixels_converts_in_parts_alike():
    c = pdfwrite.Canvas(BytesIO(), pagesize = (300, 300))
    c.setFillGradient('radial', (150, 150, 200), [(0.0, (0.9, 0.2, 0.1, 0.9)), (1.0, (0.1, 0.3, 0.8, 0.4))])
    c.rect(0, 0, 300, 300, fill = 1, stroke = 0)
    for background in ((1.0, 1.0, 1.0), None):
        parts = raster.render(c.ops, 300, 300, 4.0, 4.0, background)
        with patch.object(raster, 'THREADS', 1):
            assert np.array_equal(parts, raster.render(c.ops, 300, 300, 4.0, 4.0, background))


def _strokes_page(size, together, path, widths, alphas, dash = None, clip = None):
    """A path stroked at several widths, as one strokes op or as a stroke per width, rendered at 2x on white."""
    c = pdfwrite.Canvas(BytesIO(), pagesize = (size, size))
    c.setStrokeColorRGB(0.2, 0.5, 0.7)
    c.setLineCap(1)
    c.setLineJoin(1)
    c.setDash(dash)
    c.state['clip'] = clip
    p = c.beginPath()
    p.moveTo(*path[0])
    for x, y in path[1:]:
        p.lineTo(x, y)
    if together:
        c.drawStrokes(p, widths, alphas)
    else:
        for w, a in zip(widths, alphas):
            c.setLineWidth(w)
            c.setStrokeColorRGB(0.2, 0.5, 0.7, alpha = a)
            c.drawPath(p, fill = 0, stroke = 1)
    return c.ops, raster.render(c.ops, size, size, 2.0, 2.0, (1.0, 1.0, 1.0))


@pytest.mark.parametrize('size, dash', [(400, None), (400, (6, 3)), (40, None)])
def test_strokes_blend_once_as_strokes_painted_in_turn(size, dash):
    # A glow's strokes of one color over each other: over a large page as covered pixels, dashed through the stroke
    # pieces, and over a small box as a whole array
    wave = [(size * (0.05 + 0.9 * i / 39), size * (0.5 + 0.35 * math.sin(i / 4))) for i in range(40)]
    widths, alphas = [8.4, 5.4, 3.2, 2.0], [0.06, 0.1, 0.2, 1.0]
    ops, together = _strokes_page(size, True, wave, widths, alphas, dash)
    assert [op['kind'] for op in ops] == ['strokes']
    _, apart = _strokes_page(size, False, wave, widths, alphas, dash)
    assert np.abs(together.astype(int) - apart.astype(int)).max() <= 1 and (together < 250).sum() > 1000


def test_strokes_that_cover_most_of_their_box_or_nothing():
    # One thick segment fills most of its box, which blends as one slice; it strokes into a capsule
    seg = [(2.0, 10.0), (18.0, 10.0)]
    _, together = _strokes_page(20, True, seg, [16.0, 10.0], [0.3, 1.0])
    _, apart = _strokes_page(20, False, seg, [16.0, 10.0], [0.3, 1.0])
    assert np.abs(together.astype(int) - apart.astype(int)).max() <= 1 and (together < 250).mean() > 0.5
    blank = np.full((40, 40, 3), 255, np.uint8)
    # No alpha, a clip that leaves nothing, a hairline between sub-scanlines, and an empty path draw nothing
    away = [((30, 30, 5, 5), (1, 0, 0, 1, 0, 0))]
    for widths, alphas, clip in (([4.0], [0.0], None), ([4.0], [1.0], away), ([0.01], [1.0], None)):
        assert np.array_equal(_strokes_page(20, True, seg, widths, alphas, clip = clip)[1], blank)
    c = pdfwrite.Canvas(BytesIO(), pagesize = (20, 20))
    c.drawStrokes(c.beginPath(), [1.0], [1.0])
    assert c.ops == []


def _doc(body, size = 100):
    return ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="%d" height="%d" '
            'viewBox="0 0 %d %d">%s</svg>' % (size, size, size, size, body))


def _built(body):
    return svgin.build(_doc(body), None, (font_to_use, font_to_use + '-Bold'))


def _segments(op):
    return [seg[0] for seg in op['path']]


def test_rdkit_paths_are_drawn_as_they_stand():
    style = ('fill:none;fill-rule:evenodd;stroke:#FF0000;stroke-width:2.0px;stroke-linecap:%s;stroke-linejoin:%s;'
             'stroke-opacity:1')
    # A ring joint is an open polyline, whose closing edge used to paint a lump onto the bond
    joint = "<path d='M 10,10 L 50,10 L 50,50' style='%s' />" % (style % ('butt', 'miter'))
    op = _render_svg_to_pdf_canvas(_doc(joint), None).ops[-1]
    assert _segments(op) == ['m', 'l', 'l'] and (op['cap'], op['join']) == (0, 0) and op['line_width'] == 2.0
    # A hatch of a wedge is one segment: butt, as RDKit says, not a retraced capsule
    hatch = "<path d='M 10,10 L 50,10' style='%s' />" % (style % ('butt', 'miter'))
    assert _segments(_render_svg_to_pdf_canvas(_doc(hatch), None).ops[-1]) == ['m', 'l']
    # Only the line style named counts, and the other falls back to SVG's default
    capped = "<path d='M 10,10 L 50,10' style='fill:none;stroke:#000000;stroke-linecap:round' />"
    op = _render_svg_to_pdf_canvas(_doc(capped), None).ops[-1]
    assert (op['cap'], op['join']) == (1, 0)
    # GlycoDraw's elements have no style attribute and keep the closing rule
    plain = '<path d="M 10,10 L 50,10 L 50,50" stroke="#000000" fill="none" />'
    assert _segments(_render_svg_to_pdf_canvas(_doc(plain), None).ops[-1])[-1] == 'h'


def test_a_path_nested_in_defs_is_not_drawn():
    nested = ('<defs><g><path d="M0,0 L9,0 L9,9 Z" fill="#FF0000" id="p" /></g></defs>'
              '<rect width="5" height="5" fill="#00FF00" />')
    ops = _render_svg_to_pdf_canvas(_doc(nested), None).ops
    assert len(ops) == 1 and ops[0]['fill_rgb'] == pytest.approx((0, 1, 0))


@pytest.mark.parametrize('path, circle, end', [('M0,50 L80,50', 'cx="80" cy="50"', 55),
                                               ('M0,50 L10,50', 'cx="10" cy="50"', 10),
                                               ('M0,50 L80,50', 'cx="40" cy="50"', 80),
                                               ('M0,50 H80', 'cx="80" cy="50"', 80)])
def test_a_linkage_trims_at_an_invisible_circle(path, circle, end):
    body = '<path d="%s" stroke-width="4.0" stroke="#000000" /><circle %s r="5" fill="none" />' % (path, circle)
    op = next(o for o in _render_svg_to_pdf_canvas(_doc(body), None).ops if o['line_width'] == 4.0)
    assert max(seg[1] for seg in op['path'] if len(seg) > 1) == pytest.approx(end)


def test_a_one_command_path_stands_with_an_invisible_circle_around():
    body = '<path d="M0,50" stroke-width="4.0" stroke="#000000" /><circle cx="0" cy="50" r="5" fill="none" />'
    assert _render_svg_to_pdf_canvas(_doc(body), None).ops is not None


def test_svgin_takes_matplotlibs_universal_rule():
    rule = '<defs><style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style></defs>'
    ops = _built(rule + '<path d="M0,0 L9,9 L0,9" fill="none" stroke="#000000" />'
                 '<path d="M0,0 L9,9" fill="none" style="stroke: #000000; stroke-linejoin: miter" />').ops
    assert ops[0]['join'] == 1 and ops[1]['join'] == 0


def test_svgin_multiplies_opacity_down_a_group():
    op = _built('<g opacity="0.5"><path d="M0,0 L9,9 L0,9 Z" opacity="0.5" /><path d="M0,0 L9,9 L0,9 Z" /></g>').ops
    assert op[0]['fill_alpha'] == pytest.approx(0.25) and op[1]['fill_alpha'] == pytest.approx(0.5)


def test_svgin_reads_the_font_shorthand():
    op = _built("""<text x="5" y="5" style="font: italic bold 14.5px 'DejaVu Sans'">a</text>"""
                """<text x="5" y="5" style="font: 8px 'DejaVu Sans'">a</text>"""
                """<text x="5" y="5" style="font: oblique 9px 'Bold Face'">a</text>""").ops
    assert op[0]['size'] == 14.5 and op[0]['font'].endswith('-Bold') and op[0]['ctm'][2] != 0
    assert op[1]['size'] == 8 and not op[1]['font'].endswith('-Bold') and op[1]['ctm'][2] == 0
    assert op[2]['size'] == 9 and not op[2]['font'].endswith('-Bold') and op[2]['ctm'][2] != 0


def test_svgin_tspans_follow_the_text_before_them():
    w = lambda text, size = 10: pdfmetrics.stringWidth(text, font_to_use, size)
    # What as_standalone_svg writes for a furanose label: a tspan with no position continues the run, anchored as one
    op = _built('<text x="50" y="20" font-size="10" text-anchor="middle">D<tspan font-style="italic">f</tspan></text>'
                ).ops
    x0 = 50 - (w('D') + w('f')) / 2
    assert op[0]['ctm'][4] == pytest.approx(x0) and op[0]['ctm'][5] == pytest.approx(80) and op[0]['ctm'][2] == 0
    assert op[1]['ctm'][4] == pytest.approx(x0 + w('D')) and op[1]['ctm'][5] == pytest.approx(80)
    assert op[1]['ctm'][2] != 0
    # A lone tspan takes its parent's position, and text after a tspan carries on from it
    op = _built('<text x="10" y="20" font-size="10"><tspan>a</tspan>b</text>').ops
    assert [o['text'] for o in op] == ['a', 'b']
    assert op[1]['ctm'][4] == pytest.approx(10 + w('a')) and op[1]['ctm'][5] == 80
    # An absolute x starts a new chunk, dx and dy shift in user units or em, and a line break between tspans is no space
    op = _built('<text font-size="10" text-anchor="end">\n<tspan x="50" y="10">a</tspan>\n'
                '<tspan x="90" dy="1em" dx="2">bc</tspan></text>').ops
    assert [o['text'] for o in op] == ['a', 'bc']
    assert op[0]['ctm'][4] == pytest.approx(50 - w('a')) and op[0]['ctm'][5] == pytest.approx(90)
    assert op[1]['ctm'][4] == pytest.approx(92 - w('bc')) and op[1]['ctm'][5] == pytest.approx(80)
    # A space between words on one line stays; the ends of the text do not count
    assert [o['text'] for o in _built('<text x="0" y="9">  a <tspan>b</tspan> </text>').ops] == ['a ', 'b']
    # Anything else within a text is drawn by its own rules
    mixed = _built('<text x="0" y="9">a<circle r="3" fill="#000000" /></text>').ops
    assert {o['kind'] for o in mixed} == {'path', 'text'}
    assert not _built('<text x="0" y="9"> </text>').ops


def test_svg_output_keeps_the_miter_limit_of_png_and_pdf():
    c = pdfwrite.Canvas(None, pagesize = (50, 50))
    c.setLineJoin(0)
    p = c.beginPath()
    p.moveTo(5, 5)
    p.lineTo(40, 10)
    p.lineTo(5, 15)
    c.drawPath(p, fill = 0, stroke = 1)
    assert 'stroke-miterlimit="10"' in c.to_svg().split('\n')[1]
    c.state['join'] = 1
    c.ops[0]['join'] = 1
    assert 'miterlimit' not in c.to_svg()


def test_svg_output_is_the_same_bytes_each_time():
    def doc(color):
        c = pdfwrite.Canvas(None, pagesize = (50, 50))
        c.setFillGradient('linear', (0, 0, 50, 0), [(0, color), (1, (0, 0, 1, 1))])
        c.rect(0, 0, 50, 50, fill = 1, stroke = 0)
        return c.to_svg()
    one, two = doc((1, 0, 0, 1)), doc((0, 1, 0, 1))
    assert one == doc((1, 0, 0, 1)) and '\0' not in one
    # Pasted into one figure, different documents still keep their ids apart
    assert re.search(r'id="(\w+)"', one).group(1) != re.search(r'id="(\w+)"', two).group(1)


def _pdf(c):
    c.target = BytesIO()
    c.save()
    return c.target.getvalue()


def test_a_gradients_alpha_goes_into_its_mask():
    c = pdfwrite.Canvas(None, pagesize = (50, 50))
    c.setFillGradient('radial', (25, 25, 20), [(0, (1, 1, 1, 1)), (1, (0, 0, 0, 0.5))])
    c.setFillColorRGB(0, 0, 0, alpha = 0.6)
    c.circle(25, 25, 20, fill = 1, stroke = 0)
    data = _pdf(c)
    assert b'/C0 [0.6] /C1 [0.3]' in data and b'/ca 1 /CA 1' in data and b'/ca 0.6' not in data


def test_transparent_pages_blend_in_their_own_color_space():
    group = b'/Group << /Type /Group /S /Transparency /CS /DeviceRGB >>'
    plain = pdfwrite.Canvas(None, pagesize = (20, 20))
    plain.rect(0, 0, 9, 9, fill = 1, stroke = 0)
    assert group not in _pdf(plain)
    see_through = pdfwrite.Canvas(None, pagesize = (20, 20))
    see_through.setFillColorRGB(0, 0, 0, alpha = 0.5)
    see_through.rect(0, 0, 9, 9, fill = 1, stroke = 0)
    assert group in _pdf(see_through)
    assert group in _pdf(_canvas(shadow = True))
    graded = pdfwrite.Canvas(None, pagesize = (20, 20))
    graded.setFillGradient('linear', (0, 0, 9, 0), [(0, (1, 0, 0, 1)), (1, (0, 0, 1, 1))])
    graded.rect(0, 0, 9, 9, fill = 1, stroke = 0)
    assert group in _pdf(graded)
    for alpha, expected in ((255, False), (128, True)):
        c = pdfwrite.Canvas(None, pagesize = (20, 20))
        c.ops.append(dict(c.state, kind = 'image', w = 1, h = 1, rgb = b'\xff\0\0', alpha = bytes([alpha])))
        assert (group in _pdf(c)) == expected


def test_a_sticker_cuts_around_shapes_and_text_only():
    c = _canvas(sticker = True)
    c.ops.append(dict(c.ops[0], kind = 'image', w = 1, h = 1, rgb = b'\xff\0\0', alpha = None))
    assert _pdf(c).startswith(b'%PDF')
    assert _pixels(c, 1.0)[..., 3].any()
    c.ops.append(dict(c.ops[0], kind = 'markers', xs = [1.0], ys = [1.0]))
    assert _pdf(c).startswith(b'%PDF')


def test_fonts_are_found_in_every_font_directory_by_any_case(tmp_path):
    folder = tmp_path / 'deep' / 'er'
    folder.mkdir(parents = True)
    source = next(p for p in pdfwrite.pdfmetrics.fonts.values()).ttf.path
    shutil.copy(source, folder / 'SomeFont.TTF')
    pdfwrite._installed_fonts.cache_clear()
    try:
        with patch.object(pdfwrite, '_FONT_DIRS', [str(tmp_path / 'none'), str(tmp_path)]):
            assert pdfwrite.TTFont('Some', 'somefont.ttf').ttf.path == str(folder / 'SomeFont.TTF')
            with patch('os.walk', side_effect = AssertionError('walked twice')):
                with pytest.raises(FileNotFoundError):
                    pdfwrite.TTFont('Other', 'other.ttf')
    finally:
        pdfwrite._installed_fonts.cache_clear()
    assert 'Windows' in pdfwrite._FONT_DIRS[1] and any(d.endswith('.local/share/fonts') for d in pdfwrite._FONT_DIRS)


def test_a_glyph_set_is_subset_once():
    def page(text):
        c = pdfwrite.Canvas(None, pagesize = (50, 50))
        c.setFont(font_to_use, 10)
        c.drawString(5, 5, text)
        return _pdf(c)
    pdfwrite._font_streams.cache_clear()
    with patch.object(pdfwrite, 'subset', wraps = pdfwrite.subset) as cut:
        first = page('abc')
        assert page('abc') == first and cut.call_count == 1
        assert page('abd') != first and cut.call_count == 2


@pytest.mark.parametrize('cap, closed, expected', [(0, False, 0), (1, False, math.pi * 25), (2, False, 100),
                                                   (1, True, math.pi * 25), (2, True, 100), (0, True, 0)])
def test_a_subpath_of_one_point_gets_its_cap(cap, closed, expected):
    c = pdfwrite.Canvas(None, pagesize = (40, 40))
    c.setLineCap(cap)
    c.setLineWidth(10)
    p = c.beginPath()
    p.moveTo(20, 20)
    p.close() if closed else p.lineTo(20, 20)
    c.drawPath(p, fill = 0, stroke = 1)
    area = raster.render(c.ops, 40, 40)[..., 3].sum() / 255.0
    assert area == pytest.approx(expected, abs = 2.5)


@pytest.mark.parametrize('dash', [None, [6, 3]])
def test_a_stroke_scales_with_its_shape(dash):
    c = pdfwrite.Canvas(None, pagesize = (120, 120))
    c.scale(4, 1)
    c.setLineWidth(5)
    if dash: c.setDash(dash)
    p = c.beginPath()
    p.moveTo(15, 10)
    p.lineTo(15, 110)
    p.moveTo(5, 60)
    p.lineTo(25, 60)
    c.drawPath(p, fill = 0, stroke = 1)
    ink = raster.render(c.ops, 120, 120)[..., 3] > 127
    # 5 units across stretch to 20 pixels along x, and stay 5 along y; a dash may end before its full height
    assert ink[:40].sum(axis = 1).max() <= 21 and (dash or ink[:40].sum(axis = 1).max() >= 19)
    assert 4 <= ink[:, :40].sum(axis = 0).max() <= 6


def test_a_uniform_stroke_is_not_taken_for_a_stretched_one():
    c = pdfwrite.Canvas(None, pagesize = (60, 60))
    c.translate(30, 30)
    c.rotate(30)
    c.setLineWidth(4)
    p = c.beginPath()
    p.moveTo(-20, 0)
    p.lineTo(20, 0)
    c.drawPath(p, fill = 0, stroke = 1)
    subs = raster.flatten(c.ops[0]['path'], c.ops[0]['ctm'])
    assert raster._stroke(subs, c.ops[0], c.ops[0]['ctm'], 4)[1] is True


def _clipped_page(*clips):
    c = pdfwrite.Canvas(None, pagesize = (100, 100))
    c.setFillColorRGB(1, 0, 0)
    c.rect(0, 0, 100, 100, fill = 1, stroke = 0)
    c.saveState()
    c.setFillColorRGB(0, 0, 1)
    c.state['clip'] = list(clips)
    c.rect(0, 0, 100, 100, fill = 1, stroke = 0)
    c.restoreState()
    return raster.render(c.ops, 100, 100, background = (1, 1, 1))


def test_a_rotated_clip_cuts_along_its_own_sides():
    turn = math.radians(45)
    diamond = ((-20, -20, 40, 40), (math.cos(turn), math.sin(turn), -math.sin(turn), math.cos(turn), 50, 50))
    px = _clipped_page(diamond)
    assert tuple(px[50, 50]) == (0, 0, 255)
    # Inside the bounding box of the diamond but outside the diamond: what lay below stays
    assert tuple(px[50 - 26, 50 - 26]) == (255, 0, 0) and tuple(px[99, 99]) == (255, 0, 0)
    # Two of them: only their overlap
    other = ((-20, -20, 40, 40), (math.cos(turn), math.sin(turn), -math.sin(turn), math.cos(turn), 70, 50))
    px = _clipped_page(diamond, other)
    assert tuple(px[50, 60]) == (0, 0, 255) and tuple(px[50, 40]) == (255, 0, 0) and tuple(px[50, 80]) == (255, 0, 0)


def test_an_axis_aligned_clip_needs_no_mask():
    flip = (1.0, 0.0, 0.0, -1.0, 0.0, 100.0)
    assert raster._clip_mask(None, flip, 100, 100) is None
    assert raster._clip_mask([((0, 0, 9, 9), (1, 0, 0, 1, 0, 0))], flip, 100, 100) is None
    assert raster._clip_mask([((0, 0, 9, 9), (0, 1, -1, 0, 50, 0))], flip, 100, 100) is None


def test_the_version_is_read_when_asked_for():
    with patch('importlib.metadata.version', return_value = '9.9.9'):
        assert glycorender.__version__ == '9.9.9'
    with patch('importlib.metadata.version', side_effect = importlib.metadata.PackageNotFoundError):
        assert glycorender.__version__ == '0.0.0+local'
    with pytest.raises(AttributeError):
        glycorender.no_such_name
