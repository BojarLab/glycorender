"""Tests for glycorender: the SVG subset GlycoDraw emits, and the PNG/PDF/SVG backends."""
import base64, os, re, struct, zlib
from types import SimpleNamespace
import numpy as np
import pytest
from glycorender import pdfwrite, raster, ttf
from glycorender.render import (_render_svg_to_pdf_canvas, convert_svg_to_pdf, convert_svg_to_png, parse_color,
                                parse_path, pdf_to_svg_bytes, simple_svg_to_pdf, simple_svg_to_png)

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
        assert np.allclose(raster._box_blur(a, radius), ref, rtol = 0, atol = 1e-12)


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
        # Row r uses filter r (None, Sub, Up, Average, Paeth, then Paeth), predicted from its unfiltered neighbours
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
    assert rgba[:, :, :3].tobytes() in streams and rgba[:, :, 3].tobytes() in streams


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