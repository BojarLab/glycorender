"""Minimal PDF canvas: the subset of the reportlab API that render.py actually uses, stdlib only."""
import os, zlib, math, struct
from functools import lru_cache
from io import BytesIO
import numpy as np
from .ttf import TTF, subset

_KAPPA = 0.5522847498307936


def _utf16(code):
    if code < 0x10000:
        return (code,)
    code -= 0x10000
    return (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF))


def _fmt(v):
    if v == int(v):
        return str(int(v))
    return ('%.4f' % v).rstrip('0').rstrip('.')


def _mul(n, m):
    """n applied first, then m."""
    return (n[0]*m[0] + n[1]*m[2], n[0]*m[1] + n[1]*m[3], n[2]*m[0] + n[3]*m[2],
            n[2]*m[1] + n[3]*m[3], n[4]*m[0] + n[5]*m[2] + m[4], n[4]*m[1] + n[5]*m[3] + m[5])


def _inv(m):
    a, b, c, d, e, f = m
    det = a * d - b * c or 1e-12
    return (d/det, -b/det, -c/det, a/det, (c*f - d*e)/det, (b*e - a*f)/det)


def norm_stops(stops):
    """Sort gradient stops and clamp-pad them to span [0, 1], which is what SVG renders."""
    stops = sorted(((float(o), tuple(c) if len(c) == 4 else tuple(c) + (1.0,)) for o, c in stops), key = lambda s: s[0])
    if not stops: return [(0.0, (0.0, 0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0, 0.0))]
    if stops[0][0] > 0: stops = [(0.0, stops[0][1])] + stops
    if stops[-1][0] < 1: stops = stops + [(1.0, stops[-1][1])]
    return stops


def _stop_function(out, stops, channels):
    """PDF function interpolating the given color channels of `stops` across [0, 1]."""
    vals = ['%s' % ' '.join(_fmt(c[i]) for i in channels) for _, c in stops]
    if len(stops) < 2:
        return out.add(('<< /FunctionType 2 /Domain [0 1] /C0 [%s] /C1 [%s] /N 1 >>' % (vals[0], vals[0])).encode())
    fns = [out.add(('<< /FunctionType 2 /Domain [0 1] /C0 [%s] /C1 [%s] /N 1 >>' % (vals[i], vals[i + 1])).encode())
           for i in range(len(stops) - 1)]
    bounds, prev = [], 0.0
    for o, _ in stops[1:-1]:
        prev = max(o, prev + 1e-6)  # /Bounds must be strictly increasing
        bounds.append(prev)
    return out.add(('<< /FunctionType 3 /Domain [0 1] /Functions [%s] /Bounds [%s] /Encode [%s] >>'
                    % (' '.join('%d 0 R' % f for f in fns), ' '.join(_fmt(b) for b in bounds),
                       ' '.join(['0 1'] * len(fns)))).encode())


def _kerned_run(ttf, text):
    """Glyph string as a TJ array; a positive TJ number shifts left, so kerns are negated."""
    gids = [ttf.gid(ch) for ch in text]
    parts, chunk = [], []
    for i, g in enumerate(gids):
        chunk.append('%04X' % g)
        k = ttf.kern(g, gids[i + 1]) if i + 1 < len(gids) else 0
        if k:
            parts.append('<%s>' % ''.join(chunk))
            parts.append(_fmt(-k * 1000.0 / ttf.units_per_em))
            chunk = []
    if chunk:
        parts.append('<%s>' % ''.join(chunk))
    return '[%s]' % ' '.join(parts)


def _symbol_unit(ops):
    """The median size of the filled and stroked shapes, the symbols of a drawing, which the effects scale with."""
    sizes = []
    for op in ops:
        if op['kind'] != 'path' or not (op['fill'] and op['stroke']):
            continue
        xs = [seg[i] for seg in op['path'] for i in range(1, len(seg), 2)]
        ys = [seg[i] for seg in op['path'] for i in range(2, len(seg), 2)]
        if xs:
            sizes.append(max(max(xs) - min(xs), max(ys) - min(ys)))
    return sorted(sizes)[len(sizes) // 2] if sizes else 50.0


def shadow_params(ops, shadow):
    """Turn `shadow=True` into a spec, scaling offset and blur to the typical symbol size.

    Offsets and blur are in points, so the result is resolution independent. Positive dy is
    downwards: the key light sits above and slightly left, as illustration convention expects.
    """
    if not shadow:
        return None
    spec = dict(shadow) if isinstance(shadow, dict) else {}
    unit = _symbol_unit(ops)
    spec.setdefault('dx', 0.03 * unit)
    spec.setdefault('dy', 0.05 * unit)
    spec.setdefault('blur', 0.07 * unit)
    spec.setdefault('alpha', 0.38)
    spec.setdefault('color', (0.0, 0.0, 0.0))
    return spec


def sticker_params(ops, sticker):
    """Turn `sticker=True` into a die-cut spec: how far the cut line sits outside the ink, in points.

    `edge` adds a second, wider band underneath in `edge_color`, i.e. the thin keyline printed
    around a real die-cut sticker; `shadow` lets the cut silhouette cast the drop shadow itself.
    """
    if not sticker:
        return None
    spec = dict(sticker) if isinstance(sticker, dict) else {}
    unit = _symbol_unit(ops)
    spec.setdefault('width', 0.18 * unit)
    spec.setdefault('color', (1.0, 1.0, 1.0))
    spec.setdefault('edge', 0.0)
    spec.setdefault('edge_color', (0.110, 0.098, 0.090))
    spec.setdefault('shadow', True)
    return spec


def _grad_index(shadings, grad, bbox, alpha):
    key = (grad, bbox, round(alpha, 4))
    if key not in shadings:
        shadings[key] = len(shadings)
    return shadings[key]


def _alpha_name(alphas, value, stroking):
    key = (round(value, 4), stroking)
    if key not in alphas:
        alphas[key] = 'GS%d' % len(alphas)
    return alphas[key]


def _esc(b):
    return b.replace(b'\\', b'\\\\').replace(b'(', b'\\(').replace(b')', b'\\)')


class _Registry:
    def __init__(self):
        """Registers fonts."""
        self.fonts = {}
    def registerFont(self, font):
        self.fonts[font.name] = font
    def registerFontFamily(self, family, normal=None, bold=None, italic=None, boldItalic=None):
        pass
    def stringWidth(self, text, fontname, size):
        font = self.fonts.get(fontname)
        return font.ttf.string_width(text, size) if font else 0.0


_FONT_DIRS = [os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts'),
              os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~/AppData/Local')), 'Microsoft',
                           'Windows', 'Fonts'),
              '/Library/Fonts', '/System/Library/Fonts', os.path.expanduser('~/Library/Fonts'),
              '/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts'),
              os.path.expanduser('~/.local/share/fonts')]


@lru_cache(maxsize = None)
def _installed_fonts():
    """Every font file of the platform's font directories by lowercase name, walked once for all the names tried."""
    found = {}
    for root in _FONT_DIRS:
        for base, _, files in os.walk(root):
            for name in files:
                found.setdefault(name.lower(), os.path.join(base, name))
    return found


class TTFont:
    def __init__(self, name, filename):
        """Font class."""
        self.name = name
        if not os.path.isfile(filename):  # bare filename: search the platform font directories, as reportlab does
            filename = _installed_fonts().get(filename.lower()) or filename
            if not os.path.isfile(filename):
                raise FileNotFoundError(filename)
        self.ttf = TTF(filename)


pdfmetrics = _Registry()


class Path:
    def __init__(self):
        """A drawn path."""
        self.ops = []
    def moveTo(self, x, y):
        self.ops.append(('m', x, y))
    def lineTo(self, x, y):
        self.ops.append(('l', x, y))
    def curveTo(self, x1, y1, x2, y2, x3, y3):
        self.ops.append(('c', x1, y1, x2, y2, x3, y3))
    def close(self):
        self.ops.append(('h',))


_PDF_OP = {'m': '%s %s m', 'l': '%s %s l', 'c': '%s %s %s %s %s %s c', 'h': 'h'}


@lru_cache(maxsize = 64)
def _font_streams(ttf, kept):
    """A font's subset and ToUnicode map for a glyph set, deflated, and the length of the raw subset. Drawings keep
    using the same few glyph sets, and these streams were a third of the time a small PDF took."""
    raw = subset(ttf, kept)
    # Map only the glyphs the page uses; the whole cmap made up ~40% of a typical GlycoDraw PDF
    used = set(kept)
    pairs = sorted((g, ch) for ch, g in ttf.cmap.items() if g in used)
    cmap = ('/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /A def /CMapType 2 def\n'
            '1 begincodespacerange <0000> <FFFF> endcodespacerange\n')
    for i in range(0, len(pairs), 100):
        chunk = pairs[i:i+100]
        cmap += '%d beginbfchar\n' % len(chunk)
        cmap += ''.join('<%04X> <%s>\n' % (g, ''.join('%04X' % s for s in _utf16(ch))) for g, ch in chunk)
        cmap += 'endbfchar\n'
    cmap += 'endcmap CMapName currentdict /CMap defineresource pop end end'
    return zlib.compress(raw, 9), len(raw), zlib.compress(cmap.encode('latin-1'), 9)


class Canvas:
    def __init__(self, target, pagesize=(595.27, 841.89)):
        """Records a resolved display list; every op carries its own CTM and graphics state."""
        self.target = target
        self.width, self.height = pagesize
        self.ops = []
        self.info = {}
        self.used = {}
        self.glyphs = {}
        self.state = {'ctm': (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), 'fill_rgb': (0.0, 0.0, 0.0), 'fill_alpha': 1.0,
                      'stroke_rgb': (0.0, 0.0, 0.0), 'stroke_alpha': 1.0, 'line_width': 1.0,
                      'cap': 0, 'join': 0, 'dash': None, 'clip': None, 'even_odd': False, 'fill_grad': None}
        self.stack = []
        self._font = None
        self._size = 0
        self.shadow = None
        self.sticker = None

    # --- state ---
    def saveState(self):
        self.stack.append(dict(self.state))
    def restoreState(self):
        if self.stack: self.state = self.stack.pop()
    def setFillColorRGB(self, r, g, b, alpha=None):
        self.state['fill_rgb'] = (r, g, b)
        if alpha is not None: self.state['fill_alpha'] = alpha
    def setFillGradient(self, kind, coords, stops):
        """Gradient fill in the current user space; supersedes the flat fill until reset.

        kind is 'radial' with coords (cx, cy, r), or 'linear' with coords (x1, y1, x2, y2).
        """
        self.state['fill_grad'] = (kind, tuple(float(v) for v in coords), tuple(norm_stops(stops)))
    def setStrokeColorRGB(self, r, g, b, alpha=None):
        self.state['stroke_rgb'] = (r, g, b)
        if alpha is not None: self.state['stroke_alpha'] = alpha
    def setLineWidth(self, w):
        self.state['line_width'] = w
    def setLineCap(self, mode):
        self.state['cap'] = mode
    def setLineJoin(self, mode):
        self.state['join'] = mode
    def setDash(self, pattern=None, phase=0):
        self.state['dash'] = ([float(v) for v in pattern], float(phase)) if pattern else None
    # --- transforms ---
    def transform(self, a, b, c, d, e, f):
        m, n = self.state['ctm'], (a, b, c, d, e, f)
        self.state['ctm'] = (n[0]*m[0] + n[1]*m[2], n[0]*m[1] + n[1]*m[3], n[2]*m[0] + n[3]*m[2],
                             n[2]*m[1] + n[3]*m[3], n[4]*m[0] + n[5]*m[2] + m[4], n[4]*m[1] + n[5]*m[3] + m[5])
    def translate(self, dx, dy):
        self.transform(1, 0, 0, 1, dx, dy)
    def scale(self, sx, sy):
        self.transform(sx, 0, 0, sy, 0, 0)
    def rotate(self, deg):
        r = math.radians(deg)
        self.transform(math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)
    # --- geometry ---
    def beginPath(self):
        return Path()
    def drawPath(self, path, fill=0, stroke=1):
        if not path.ops: return
        op = dict(self.state)
        op.update(kind='path', path=list(path.ops), fill=bool(fill), stroke=bool(stroke))
        self.ops.append(op)
    def rect(self, x, y, width, height, fill=0, stroke=1):
        p = Path()
        p.moveTo(x, y)
        p.lineTo(x + width, y)
        p.lineTo(x + width, y + height)
        p.lineTo(x, y + height)
        p.close()
        self.drawPath(p, fill=fill, stroke=stroke)
    def circle(self, cx, cy, r, fill=0, stroke=1):
        self.ellipse(cx - r, cy - r, cx + r, cy + r, fill=fill, stroke=stroke)
    def ellipse(self, x1, y1, x2, y2, fill=0, stroke=1):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rx, ry = abs(x2 - x1) / 2.0, abs(y2 - y1) / 2.0
        ox, oy = rx * _KAPPA, ry * _KAPPA
        p = Path()
        p.moveTo(cx - rx, cy)
        p.curveTo(cx - rx, cy + oy, cx - ox, cy + ry, cx, cy + ry)
        p.curveTo(cx + ox, cy + ry, cx + rx, cy + oy, cx + rx, cy)
        p.curveTo(cx + rx, cy - oy, cx + ox, cy - ry, cx, cy - ry)
        p.curveTo(cx - ox, cy - ry, cx - rx, cy - oy, cx - rx, cy)
        p.close()
        self.drawPath(p, fill=fill, stroke=stroke)
    # --- text ---
    def setFont(self, name, size):
        self._font, self._size = pdfmetrics.fonts[name], size
        self.used[name] = self._font
    def drawString(self, x, y, text, charSpace=0):
        if not text or self._font is None: return
        self.glyphs.setdefault(self._font.name, set()).update(self._font.ttf.gid(ch) for ch in text)
        op = dict(self.state)
        op.update(kind='text', x=x, y=y, text=text, font=self._font.name, ttf=self._font.ttf,
                  size=self._size, char_space=charSpace)
        self.ops.append(op)

    def drawMarkers(self, path, xs, ys, fill=1, stroke=0):
        """One path, written around its own origin, painted with the current state at each (x, y). For PDF output only,
        where the path is stored once as a form, as matplotlib does: to_png and to_svg do not take this op."""
        if not path.ops: return
        op = dict(self.state)
        op.update(kind='markers', path=list(path.ops), xs=xs, ys=ys, fill=bool(fill), stroke=bool(stroke))
        self.ops.append(op)

    def drawStrokes(self, path, widths, alphas):
        """One path stroked in the current stroke color at each width with its alpha, from the bottom up, as a glow
        around a line. For raster output only, which blends them once: to_pdf and to_svg do not take this op."""
        if not path.ops: return
        op = dict(self.state)
        op.update(kind='strokes', path=list(path.ops), widths=list(widths), alphas=list(alphas), fill=False,
                  stroke=True)
        self.ops.append(op)

    def drawImage(self, png):
        """PNG bytes onto the unit square of the current user space, first row at the top, as PDF images go.

        Covers what matplotlib embeds for rasterized artists (colorbars, imshow): 8-bit, non-interlaced
        gray, gray+alpha, RGB or RGBA. Anything else is skipped rather than drawn wrong.
        """
        pos, idat, head = 8, [], None
        while png[:8] == b'\x89PNG\r\n\x1a\n' and pos + 8 <= len(png):
            n, tag = struct.unpack('>I4s', png[pos:pos + 8])
            if tag == b'IHDR':
                head = struct.unpack('>IIBBBBB', png[pos + 8:pos + 21])
            elif tag == b'IDAT':
                idat.append(png[pos + 8:pos + 8 + n])
            pos += 12 + n
        bpp = {0: 1, 2: 3, 4: 2, 6: 4}.get(head[3]) if head and head[2] == 8 and not head[6] else None
        if not bpp: return
        w, h = head[0], head[1]
        raw, stride = zlib.decompress(b''.join(idat)), w * bpp
        prev, rows = bytearray(stride), []
        for r in range(h):
            f, line = raw[r * (stride + 1)], bytearray(raw[r * (stride + 1) + 1:(r + 1) * (stride + 1)])
            # Up and None are whole-row operations; Sub runs along each channel; Average and Paeth chain byte by byte.
            # Up on numpy: a large figure's shadow, 26 million bytes, took 2.4 s through a generator and takes 0.03 s
            if f == 2: line = bytearray((np.frombuffer(line, np.uint8) + np.frombuffer(prev, np.uint8)).tobytes())
            if f == 1:
                line = bytearray(np.add.accumulate(np.frombuffer(line, np.uint8).reshape(-1, bpp), axis=0,
                                                   dtype=np.uint8).tobytes())
            for i in range(stride) if f in (3, 4) else ():
                a = line[i - bpp] if i >= bpp else 0
                if f == 3:
                    line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
                else:
                    b, c = prev[i], prev[i - bpp] if i >= bpp else 0
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
            rows.append(line)
            prev = line
        px, rgb = b''.join(rows), bytearray(w * h * 3)
        for k in range(3):
            rgb[k::3] = px[(k if bpp > 2 else 0)::bpp]
        op = dict(self.state)
        op.update(kind = 'image', w = w, h = h, rgb = bytes(rgb), alpha = px[bpp - 1::bpp] if bpp in (2, 4) else None)
        self.ops.append(op)

    # --- metadata ---
    def setTitle(self, v):
        self.info['Title'] = v
    def setAuthor(self, v):
        self.info['Author'] = v
    def setSubject(self, v):
        self.info['Subject'] = v
    def setKeywords(self, v):
        self.info['Keywords'] = v

    # --- backends ---
    def _effects(self):
        """Resolved (shadow, sticker) specs; a die cut casts the shadow itself unless told otherwise."""
        sticker = sticker_params(self.ops, self.sticker)
        return shadow_params(self.ops, self.shadow or (sticker['shadow'] if sticker else False)), sticker

    def to_png(self, scale_x = 1.0, scale_y = 1.0, background = None, texts = ()):
        from . import raster
        shadow, sticker = self._effects()
        return raster.encode_png(
            raster.render(self.ops, self.width, self.height, scale_x, scale_y, background, shadow, sticker), texts,
            (72.0 * scale_x, 72.0 * scale_y))

    def to_svg(self):
        from .svgout import emit
        return emit(self.ops, self.width, self.height, *self._effects())

    # --- PDF output ---
    def _path_bbox(self, path, grad):
        kind, coords = grad[0], grad[1]
        if kind == 'radial':
            cx, cy, r = coords
            xs, ys = [cx - r, cx + r], [cy - r, cy + r]
        else:
            xs, ys = [coords[0], coords[2]], [coords[1], coords[3]]
        for seg in path:
            for i in range(1, len(seg), 2):
                xs.append(seg[i])
                ys.append(seg[i + 1])
        return (min(xs), min(ys), max(xs), max(ys))
    def _content(self, alphas, shadings, images, forms):
        buf = []
        for op in self.ops:
            buf.append('q')
            cur = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            for box, ctm in op.get('clip') or ():
                buf.append('%s %s %s %s %s %s cm' % tuple(_fmt(v) for v in _mul(ctm, _inv(cur))))
                buf.append('%s %s %s %s re W n' % tuple(_fmt(v) for v in box))
                cur = ctm
            buf.append('%s %s %s %s %s %s cm' % tuple(_fmt(v) for v in _mul(op['ctm'], _inv(cur))))
            if op['kind'] == 'path' or op['kind'] == 'markers':
                grad = op.get('fill_grad')
                path = [_PDF_OP[seg[0]] % tuple(_fmt(v) for v in seg[1:]) if len(seg) > 1 else _PDF_OP[seg[0]]
                        for seg in op['path']]
                eo = '*' if op.get('even_odd') else ''
                if op['kind'] == 'markers':
                    # A form inherits the color, alpha, and line width set below; its box depends on that line width
                    paint = ('B' + eo) if (op['fill'] and op['stroke']) else ('f' + eo) if op['fill'] else 'S'
                    name = forms.setdefault((tuple(op['path']), paint, op['line_width'] if op['stroke'] else 0.0),
                                            'M%d' % len(forms))
                    path = ['q 1 0 0 1 %s %s cm /%s Do Q' % (_fmt(x), _fmt(y), name) for x, y in zip(op['xs'], op['ys'])]
                if op['fill'] and grad:  # paint the shading through the shape used as a clip
                    idx = _grad_index(shadings, grad, self._path_bbox(op['path'], grad), op['fill_alpha'])
                    buf.append('q')
                    buf.extend(path)
                    buf.append('W%s n' % eo)
                    buf.append('/GSh%d gs /Sh%d sh Q' % (idx, idx))
                solid = op['fill'] and not grad
                if not (solid or op['stroke']):
                    buf.append('Q')
                    continue
                if solid:
                    buf.append('/%s gs' % _alpha_name(alphas, op['fill_alpha'], False))
                    buf.append('%s %s %s rg' % tuple(_fmt(v) for v in op['fill_rgb']))
                if op['stroke']:
                    buf.append('/%s gs' % _alpha_name(alphas, op['stroke_alpha'], True))
                    buf.append('%s %s %s RG' % tuple(_fmt(v) for v in op['stroke_rgb']))
                    buf.append('%s w %d J %d j' % (_fmt(op['line_width']), op['cap'], op['join']))
                    if op['dash']:
                        buf.append('[%s] %s d' % (' '.join(_fmt(v) for v in op['dash'][0]), _fmt(op['dash'][1])))
                buf.extend(path)
                if op['kind'] == 'path':
                    buf.append(('B' + eo) if (solid and op['stroke']) else ('f' + eo) if solid else 'S')
            elif op['kind'] == 'image':
                buf.append('/%s gs /Im%d Do' % (_alpha_name(alphas, op['fill_alpha'], False), len(images)))
                images.append(op)
            else:
                buf.append('/%s gs' % _alpha_name(alphas, op['fill_alpha'], False))
                buf.append('%s %s %s rg' % tuple(_fmt(v) for v in op['fill_rgb']))
                buf.append('BT /%s %s Tf %s Tc 1 0 0 1 %s %s Tm %s TJ ET'
                           % (op['font'], _fmt(op['size']), _fmt(op['char_space']), _fmt(op['x']), _fmt(op['y']),
                              _kerned_run(op['ttf'], op['text'])))
            buf.append('Q')
        return '\n'.join(buf)

    def _sticker_content(self, alphas, spec):
        """The die-cut layer: every shape re-emitted in flat color, fattened by a round stroke."""
        buf = []
        layers = [(spec['width'] + spec['edge'], spec['edge_color'])] if spec['edge'] > 0 else []
        for grow, color in layers + [(spec['width'], spec['color'])]:
            for op in self.ops:
                # Text and shapes have a silhouette to cut around; an image or a form of markers has none of its own
                if op['kind'] not in ('path', 'text') or op['kind'] == 'path' and not (op['fill'] or op['stroke']):
                    continue
                buf.append('q')
                cur = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                for box, ctm in op.get('clip') or ():
                    buf.append('%s %s %s %s %s %s cm' % tuple(_fmt(v) for v in _mul(ctm, _inv(cur))))
                    buf.append('%s %s %s %s re W n' % tuple(_fmt(v) for v in box))
                    cur = ctm
                buf.append('%s %s %s %s %s %s cm' % tuple(_fmt(v) for v in _mul(op['ctm'], _inv(cur))))
                buf.append('/%s gs /%s gs' % (_alpha_name(alphas, 1.0, False), _alpha_name(alphas, 1.0, True)))
                buf.append('%s %s %s rg %s %s %s RG' % (tuple(_fmt(v) for v in color) * 2))
                is_path = op['kind'] == 'path'
                buf.append('%s w 1 J 1 j' % _fmt((op['line_width'] if is_path and op['stroke'] else 0.0) + 2 * grow))
                if is_path:
                    buf.extend(_PDF_OP[seg[0]] % tuple(_fmt(v) for v in seg[1:]) if len(seg) > 1 else _PDF_OP[seg[0]]
                               for seg in op['path'])
                    buf.append(('B*' if op.get('even_odd') else 'B') if op['fill'] else 'S')
                else:
                    buf.append('BT /%s %s Tf %s Tc 2 Tr 1 0 0 1 %s %s Tm %s TJ ET'
                               % (op['font'], _fmt(op['size']), _fmt(op['char_space']), _fmt(op['x']), _fmt(op['y']),
                                  _kerned_run(op['ttf'], op['text'])))
                buf.append('Q')
        return '\n'.join(buf)
    def _shadow_object(self, out, spec, sticker):
        """Flat shadow color behind a grayscale soft mask; a blurred shadow cannot stay vector."""
        from . import raster
        w, h, rows = raster.shadow_alpha(self.ops, self.width, self.height, spec, sticker)
        packed = zlib.compress(rows, 9)
        smask = out.add(('<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray '
                         '/BitsPerComponent 8 /Filter /FlateDecode /DecodeParms << /Predictor 15 /Columns %d >> '
                         '/Length %d >>\nstream\n' % (w, h, w, len(packed))).encode() + packed + b'\nendstream')
        # The color is one flat value, so the image under the mask is a one-entry palette at a bit per pixel
        body = zlib.compress(bytes((w + 7) // 8 * h), 9)
        color = ''.join('%02X' % int(round(v * 255)) for v in spec['color'])
        return out.add(('<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace [/Indexed /DeviceRGB 0 '
                        '<%s>] /BitsPerComponent 1 /Filter /FlateDecode /SMask %d 0 R /Length %d >>\nstream\n'
                        % (w, h, color, smask, len(body))).encode() + body + b'\nendstream')
    def _shading_objects(self, out, grad, bbox, alpha):
        """Radial shading plus the luminosity soft mask that carries its per-stop alpha."""
        kind, geo, stops = grad
        if kind == 'radial':
            cx, cy, r = geo
            stype, coords = 3, '[%s %s 0 %s %s %s]' % (_fmt(cx), _fmt(cy), _fmt(cx), _fmt(cy), _fmt(r))
        else:
            stype, coords = 2, '[%s %s %s %s]' % tuple(_fmt(v) for v in geo)
        sh = out.add(('<< /ShadingType %d /ColorSpace /DeviceRGB /Coords %s /Function %d 0 R /Extend [true true] >>'
                      % (stype, coords, _stop_function(out, stops, (0, 1, 2)))).encode())
        # The op's alpha goes into the mask, not next to it as /ca, which MuPDF applies inside the mask as well as out
        alpha_sh = out.add(
            ('<< /ShadingType %d /ColorSpace /DeviceGray /Coords %s /Function %d 0 R /Extend [true true] >>'
             % (stype, coords, _stop_function(out, [(o, (c[3] * alpha,) * 4) for o, c in stops], (3,)))).encode())
        body = b'/S0 sh'
        form = out.add(('<< /Type /XObject /Subtype /Form /BBox [%s %s %s %s] '
                        '/Group << /Type /Group /S /Transparency /CS /DeviceGray >> '
                        '/Resources << /Shading << /S0 %d 0 R >> >> /Length %d >>\nstream\n'
                        % (_fmt(bbox[0]), _fmt(bbox[1]), _fmt(bbox[2]), _fmt(bbox[3]), alpha_sh, len(body))).encode()
                       + body + b'\nendstream')
        return sh, '<< /Type /ExtGState /SMask << /S /Luminosity /G %d 0 R /BC [0] >> /ca 1 /CA 1 >>' % form
    def _font_objects(self, out, name, font):
        ttf = font.ttf
        scale = 1000.0 / ttf.units_per_em
        kept = sorted(g for g in self.glyphs.get(name, set()) if g < ttf.num_glyphs)
        # A subset must be named TAG+Font (ISO 32000-1, 9.6.4), or readers take it for the complete font; the tag
        # is derived from the glyph set, so the output stays reproducible and different subsets never share a name
        crc = zlib.crc32(str(kept).encode())
        base = ''.join(chr(65 + crc // 26 ** i % 26) for i in range(6)) + '+' + name
        packed, raw_length, packed_cmap = _font_streams(ttf, tuple(kept))
        file_ref = out.add(b'<< /Length %d /Length1 %d /Filter /FlateDecode >>\nstream\n' % (len(packed), raw_length)
                           + packed + b'\nendstream')
        flags = 4 | (1 if ttf.mac_style & 2 else 0) << 6
        desc = out.add(('<< /Type /FontDescriptor /FontName /%s /Flags %d /FontBBox [%d %d %d %d] '
                        '/ItalicAngle %s /Ascent %d /Descent %d /CapHeight %d /StemV 80 /FontFile2 %d 0 R >>'
                        % (base, flags, *[int(v * scale) for v in ttf.bbox], _fmt(ttf.italic_angle),
                           int(ttf.ascent * scale), int(ttf.descent * scale), int(ttf.cap_height * scale),
                           file_ref)).encode())
        widths = '[%s]' % ' '.join('%d [%d]' % (g, int(round(ttf.width(g)))) for g in kept)
        cid = out.add(('<< /Type /Font /Subtype /CIDFontType2 /BaseFont /%s /CIDSystemInfo '
                       '<< /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor %d 0 R '
                       '/DW 1000 /W %s /CIDToGIDMap /Identity >>' % (base, desc, widths)).encode())
        tou = out.add(b'<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(packed_cmap) + packed_cmap + b'\nendstream')
        return out.add(('<< /Type /Font /Subtype /Type0 /BaseFont /%s /Encoding /Identity-H '
                        '/DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>' % (base, cid, tou)).encode())
    def save(self):
        out = _Writer()
        alphas, shadings, images, forms = {}, {}, [], {}
        body = self._content(alphas, shadings, images, forms)
        shadow, sticker = self._effects()
        if sticker:
            body = self._sticker_content(alphas, sticker) + '\n' + body
        xobj = ''
        from .raster import deflate_rows
        for i, op in enumerate(images):
            smask = ''
            # Rows filtered as in a PNG, which a soft shadow or a smooth heatmap packs far tighter with
            if op['alpha'] is not None and op['alpha'].count(255) < len(op['alpha']):
                packed = deflate_rows(op['alpha'], op['h'])
                smask = '/SMask %d 0 R ' % out.add(
                    ('<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray /BitsPerComponent 8 '
                     '/Filter /FlateDecode /DecodeParms << /Predictor 15 /Columns %d >> /Length %d >>\nstream\n'
                     % (op['w'], op['h'], op['w'], len(packed))).encode() + packed + b'\nendstream')
            packed = deflate_rows(op['rgb'], op['h'])
            xobj += '/Im%d %d 0 R ' % (i, out.add(
                ('<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 '
                 '%s/Filter /FlateDecode /DecodeParms << /Predictor 15 /Colors 3 /Columns %d >> /Length %d >>\nstream\n'
                 % (op['w'], op['h'], smask, op['w'], len(packed))).encode() + packed + b'\nendstream'))
        for (path, paint, width), name in forms.items():
            # Past the path's own points by as far as a miter at the default limit can reach
            xs = [seg[i] for seg in path for i in range(1, len(seg), 2)]
            ys = [seg[i] for seg in path for i in range(2, len(seg), 2)]
            bbox = ' '.join(_fmt(v) for v in (min(xs) - 5 * width, min(ys) - 5 * width, max(xs) + 5 * width,
                                              max(ys) + 5 * width))
            draw = [_PDF_OP[seg[0]] % tuple(_fmt(v) for v in seg[1:]) if len(seg) > 1 else _PDF_OP[seg[0]]
                    for seg in path]
            packed = zlib.compress('\n'.join(draw + [paint]).encode('latin-1'), 9)
            xobj += '/%s %d 0 R ' % (name, out.add(('<< /Type /XObject /Subtype /Form /BBox [%s] /Filter /FlateDecode '
                                                    '/Length %d >>\nstream\n' % (bbox, len(packed))).encode()
                                                   + packed + b'\nendstream'))
        if shadow:
            xobj += '/Shadow %d 0 R' % self._shadow_object(out, shadow, sticker)
            body = 'q %s 0 0 %s 0 0 cm /Shadow Do Q\n' % (_fmt(self.width), _fmt(self.height)) + body
        # Level 9 gains a percent or two over 6 at twice the time, which only a very large page would notice
        content = zlib.compress(body.encode('latin-1'), 9 if len(body) < 1 << 20 else 6)
        stream = out.add(b'<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(content) + content + b'\nendstream')
        fonts = ' '.join('/%s %d 0 R' % (n, self._font_objects(out, n, f)) for n, f in self.used.items())
        gs = [('/%s << /Type /ExtGState /%s %s >>' % (n, 'CA' if k[1] else 'ca', _fmt(k[0]))) for k, n in
              alphas.items()]
        sh = []
        for (grad, bbox, alpha), idx in shadings.items():
            ref, ext = self._shading_objects(out, grad, bbox, alpha)
            sh.append('/Sh%d %d 0 R' % (idx, ref))
            gs.append('/GSh%d %s' % (idx, ext))
        res = '<< /Font << %s >> /ExtGState << %s >> /Shading << %s >> /XObject << %s >> >>' % (
            fonts, ' '.join(gs), ' '.join(sh), xobj)
        pages_ref = out.reserve()
        # Transparent content on a page of its own color space, as matplotlib does, or a reader may blend it in another
        # one, such as Acrobat's working CMYK, and shift its colors
        clear = (shadow or shadings or any(k[0] < 1 for k in alphas) or
                 any(op['alpha'] is not None and op['alpha'].count(255) < len(op['alpha']) for op in images))
        group = ' /Group << /Type /Group /S /Transparency /CS /DeviceRGB >>' if clear else ''
        page = out.add(('<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %s %s] /Resources %s /Contents %d 0 R%s >>'
                        % (pages_ref, _fmt(self.width), _fmt(self.height), res, stream, group)).encode())
        out.set(pages_ref, ('<< /Type /Pages /Kids [%d 0 R] /Count 1 >>' % page).encode())
        root = out.add(('<< /Type /Catalog /Pages %d 0 R >>' % pages_ref).encode())
        info = None
        if self.info:
            # Non-ASCII text strings go out as UTF-16BE with a BOM, so α, β, and → survive instead of becoming '?'
            info = out.add(b'<< ' + b' '.join(
                b'/%s (%s)' % (k.encode(), _esc(v.encode('ascii'))) if v.isascii()
                else b'/%s <FEFF%s>' % (k.encode(), v.encode('utf-16-be').hex().upper().encode())
                for k, v in self.info.items()) + b' >>')
        data = out.render(root, info)
        if hasattr(self.target, 'write'):
            self.target.write(data)
        else:
            with open(str(self.target), 'wb') as fh:
                fh.write(data)


class _Writer:
    def __init__(self):
        """Writes out the pdf."""
        self.objs = []
    def reserve(self):
        self.objs.append(None)
        return len(self.objs)
    def set(self, num, body):
        self.objs[num - 1] = body
    def add(self, body):
        self.objs.append(body)
        return len(self.objs)
    def render(self, root, info):
        out = BytesIO()
        out.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
        offsets = []
        for i, body in enumerate(self.objs, 1):
            offsets.append(out.tell())
            out.write(b'%d 0 obj\n' % i + body + b'\nendobj\n')
        xref = out.tell()
        out.write(b'xref\n0 %d\n0000000000 65535 f \n' % (len(self.objs) + 1))
        for off in offsets:
            out.write(b'%010d 00000 n \n' % off)
        trailer = b'trailer\n<< /Size %d /Root %d 0 R' % (len(self.objs) + 1, root)
        if info:
            trailer += b' /Info %d 0 R' % info
        out.write(trailer + b' >>\nstartxref\n%d\n%%%%EOF\n' % xref)
        return out.getvalue()
