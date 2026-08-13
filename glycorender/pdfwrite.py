"""Minimal PDF canvas: the subset of the reportlab API that render.py actually uses, stdlib only."""
import os, zlib, math
from io import BytesIO
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
    """PDF function interpolating the given colour channels of `stops` across [0, 1]."""
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
        self.fonts = {}
    def registerFont(self, font):
        self.fonts[font.name] = font
    def registerFontFamily(self, family, normal=None, bold=None, italic=None, boldItalic=None):
        pass
    def stringWidth(self, text, fontname, size):
        font = self.fonts.get(fontname)
        return font.ttf.string_width(text, size) if font else 0.0


_FONT_DIRS = [os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts'),
              '/Library/Fonts', '/System/Library/Fonts', os.path.expanduser('~/Library/Fonts'),
              '/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts')]


class TTFont:
    def __init__(self, name, filename):
        self.name = name
        if not os.path.isfile(filename):  # bare filename: search the platform font directories, as reportlab does
            for root in _FONT_DIRS:
                for base, _, files in os.walk(root) if os.path.isdir(root) else ():
                    hit = {f.lower(): f for f in files}.get(filename.lower())
                    if hit:
                        filename = os.path.join(base, hit)
                        break
                if os.path.isfile(filename):
                    break
            else:
                raise FileNotFoundError(filename)
        self.ttf = TTF(filename)


pdfmetrics = _Registry()


class Path:
    def __init__(self):
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


class Canvas:
    """Records a resolved display list; every op carries its own CTM and graphics state."""
    def __init__(self, target, pagesize=(595.27, 841.89)):
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
    # --- state ---
    def saveState(self):
        self.stack.append(dict(self.state))
    def restoreState(self):
        if self.stack: self.state = self.stack.pop()
    def setFillColorRGB(self, r, g, b, alpha=None):
        self.state['fill_rgb'] = (r, g, b)
        if alpha is not None: self.state['fill_alpha'] = alpha
    def setFillGradient(self, cx, cy, r, stops):
        """Radial gradient fill in the current user space; supersedes the flat fill until reset."""
        self.state['fill_grad'] = (float(cx), float(cy), float(r), tuple(norm_stops(stops)))
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
    def to_png(self, scale_x = 1.0, scale_y = 1.0, background = (1.0, 1.0, 1.0), texts = ()):
        from . import raster
        return raster.encode_png(raster.render(self.ops, self.width, self.height, scale_x, scale_y, background),
                                 texts)

    def to_svg(self):
        from .svgout import emit
        return emit(self.ops, self.width, self.height)

    # --- PDF output ---
    def _path_bbox(self, path, cx, cy, r):
        xs = [cx - r, cx + r]
        ys = [cy - r, cy + r]
        for seg in path:
            for i in range(1, len(seg), 2):
                xs.append(seg[i])
                ys.append(seg[i + 1])
        return (min(xs), min(ys), max(xs), max(ys))
    def _content(self, alphas, shadings):
        buf = []
        for op in self.ops:
            buf.append('q')
            cur = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            for box, ctm in op.get('clip') or ():
                buf.append('%s %s %s %s %s %s cm' % tuple(_fmt(v) for v in _mul(_inv(cur), ctm)))
                buf.append('%s %s %s %s re W n' % tuple(_fmt(v) for v in box))
                cur = ctm
            buf.append('%s %s %s %s %s %s cm' % tuple(_fmt(v) for v in _mul(_inv(cur), op['ctm'])))
            if op['kind'] == 'path':
                grad = op.get('fill_grad')
                path = [_PDF_OP[seg[0]] % tuple(_fmt(v) for v in seg[1:]) if len(seg) > 1 else _PDF_OP[seg[0]]
                        for seg in op['path']]
                eo = '*' if op.get('even_odd') else ''
                if op['fill'] and grad:  # paint the shading through the shape used as a clip
                    idx = _grad_index(shadings, grad, self._path_bbox(op['path'], *grad[:3]), op['fill_alpha'])
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
                buf.append(('B' + eo) if (solid and op['stroke']) else ('f' + eo) if solid else 'S')
            else:
                buf.append('/%s gs' % _alpha_name(alphas, op['fill_alpha'], False))
                buf.append('%s %s %s rg' % tuple(_fmt(v) for v in op['fill_rgb']))
                gids = ''.join('%04X' % op['ttf'].gid(ch) for ch in op['text'])
                buf.append('BT /%s %s Tf %s Tc 1 0 0 1 %s %s Tm <%s> Tj ET'
                           % (op['font'], _fmt(op['size']), _fmt(op['char_space']), _fmt(op['x']), _fmt(op['y']),
                              gids))
            buf.append('Q')
        return '\n'.join(buf)

    def _shading_objects(self, out, grad, bbox, alpha):
        """Radial shading plus the luminosity soft mask that carries its per-stop alpha."""
        cx, cy, r, stops = grad
        coords = '[%s %s 0 %s %s %s]' % (_fmt(cx), _fmt(cy), _fmt(cx), _fmt(cy), _fmt(r))
        sh = out.add(('<< /ShadingType 3 /ColorSpace /DeviceRGB /Coords %s /Function %d 0 R /Extend [true true] >>'
                      % (coords, _stop_function(out, stops, (0, 1, 2)))).encode())
        alpha_sh = out.add(('<< /ShadingType 3 /ColorSpace /DeviceGray /Coords %s /Function %d 0 R /Extend [true true] >>'
                            % (coords, _stop_function(out, stops, (3,)))).encode())
        body = b'/S0 sh'
        form = out.add(('<< /Type /XObject /Subtype /Form /BBox [%s %s %s %s] '
                        '/Group << /Type /Group /S /Transparency /CS /DeviceGray >> '
                        '/Resources << /Shading << /S0 %d 0 R >> >> /Length %d >>\nstream\n'
                        % (_fmt(bbox[0]), _fmt(bbox[1]), _fmt(bbox[2]), _fmt(bbox[3]), alpha_sh, len(body))).encode()
                       + body + b'\nendstream')
        return sh, ('<< /Type /ExtGState /SMask << /S /Luminosity /G %d 0 R /BC [0] >> /ca %s /CA %s >>'
                    % (form, _fmt(alpha), _fmt(alpha)))
    def _font_objects(self, out, name, font):
        ttf = font.ttf
        scale = 1000.0 / ttf.units_per_em
        raw = subset(ttf, self.glyphs.get(name, set()))
        packed = zlib.compress(raw, 9)
        file_ref = out.add(b'<< /Length %d /Length1 %d /Filter /FlateDecode >>\nstream\n' % (len(packed), len(raw))
                           + packed + b'\nendstream')
        flags = 4 | (1 if ttf.mac_style & 2 else 0) << 6
        desc = out.add(('<< /Type /FontDescriptor /FontName /%s /Flags %d /FontBBox [%d %d %d %d] '
                        '/ItalicAngle %s /Ascent %d /Descent %d /CapHeight %d /StemV 80 /FontFile2 %d 0 R >>'
                        % (name, flags, *[int(v * scale) for v in ttf.bbox], _fmt(ttf.italic_angle),
                           int(ttf.ascent * scale), int(ttf.descent * scale), int(ttf.cap_height * scale),
                           file_ref)).encode())
        run, start = [], None
        for gid in range(ttf.num_glyphs):
            w = int(round(ttf.width(gid)))
            if start is None:
                start, run = gid, [w]
            elif gid == start + len(run):
                run.append(w)
        widths = '[%d [%s]]' % (start, ' '.join(str(v) for v in run))
        cid = out.add(('<< /Type /Font /Subtype /CIDFontType2 /BaseFont /%s /CIDSystemInfo '
                       '<< /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor %d 0 R '
                       '/DW 1000 /W %s /CIDToGIDMap /Identity >>' % (name, desc, widths)).encode())
        pairs = sorted((g, ch) for ch, g in ttf.cmap.items() if g < ttf.num_glyphs)
        cmap = ('/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /A def /CMapType 2 def\n'
                '1 begincodespacerange <0000> <FFFF> endcodespacerange\n')
        for i in range(0, len(pairs), 100):
            chunk = pairs[i:i+100]
            cmap += '%d beginbfchar\n' % len(chunk)
            cmap += ''.join('<%04X> <%s>\n' % (g, ''.join('%04X' % s for s in _utf16(ch))) for g, ch in chunk)
            cmap += 'endbfchar\n'
        cmap += 'endcmap CMapName currentdict /CMap defineresource pop end end'
        packed_cmap = zlib.compress(cmap.encode('latin-1'), 9)
        tou = out.add(b'<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(packed_cmap) + packed_cmap + b'\nendstream')
        return out.add(('<< /Type /Font /Subtype /Type0 /BaseFont /%s /Encoding /Identity-H '
                        '/DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>' % (name, cid, tou)).encode())
    def save(self):
        out = _Writer()
        alphas, shadings = {}, {}
        content = zlib.compress(self._content(alphas, shadings).encode('latin-1'), 9)
        stream = out.add(b'<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(content) + content + b'\nendstream')
        fonts = ' '.join('/%s %d 0 R' % (n, self._font_objects(out, n, f)) for n, f in self.used.items())
        gs = [('/%s << /Type /ExtGState /%s %s >>' % (n, 'CA' if k[1] else 'ca', _fmt(k[0]))) for k, n in
              alphas.items()]
        sh = []
        for (grad, bbox, alpha), idx in shadings.items():
            ref, ext = self._shading_objects(out, grad, bbox, alpha)
            sh.append('/Sh%d %d 0 R' % (idx, ref))
            gs.append('/GSh%d %s' % (idx, ext))
        res = '<< /Font << %s >> /ExtGState << %s >> /Shading << %s >> >>' % (fonts, ' '.join(gs), ' '.join(sh))
        pages_ref = out.reserve()
        page = out.add(('<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %s %s] /Resources %s /Contents %d 0 R >>'
                        % (pages_ref, _fmt(self.width), _fmt(self.height), res, stream)).encode())
        out.set(pages_ref, ('<< /Type /Pages /Kids [%d 0 R] /Count 1 >>' % page).encode())
        root = out.add(('<< /Type /Catalog /Pages %d 0 R >>' % pages_ref).encode())
        info = None
        if self.info:
            info = out.add(b'<< ' + b' '.join(b'/%s (%s)' % (k.encode(), _esc(v.encode('latin-1', 'replace')))
                                              for k, v in self.info.items()) + b' >>')
        data = out.render(root, info)
        if hasattr(self.target, 'write'):
            self.target.write(data)
        else:
            with open(str(self.target), 'wb') as fh:
                fh.write(data)


class _Writer:
    def __init__(self):
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
