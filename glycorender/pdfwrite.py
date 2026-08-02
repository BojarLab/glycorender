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
        self.ops.append('%s %s m' % (_fmt(x), _fmt(y)))
    def lineTo(self, x, y):
        self.ops.append('%s %s l' % (_fmt(x), _fmt(y)))
    def curveTo(self, x1, y1, x2, y2, x3, y3):
        self.ops.append('%s %s %s %s %s %s c' % tuple(_fmt(v) for v in (x1, y1, x2, y2, x3, y3)))
    def close(self):
        self.ops.append('h')


class Canvas:
    def __init__(self, target, pagesize=(595.27, 841.89)):
        self.target = target
        self.width, self.height = pagesize
        self.buf = []
        self.info = {}
        self.used = {}
        self.glyphs = {}
        self.alphas = {}
        self._font = None
        self._size = 0
    # --- state ---
    def saveState(self):
        self.buf.append('q')
    def restoreState(self):
        self.buf.append('Q')
    def _alpha(self, value, stroking):
        key = (round(value, 4), stroking)
        name = self.alphas.get(key)
        if name is None:
            name = 'GS%d' % len(self.alphas)
            self.alphas[key] = name
        self.buf.append('/%s gs' % name)
    def setFillColorRGB(self, r, g, b, alpha=None):
        if alpha is not None:
            self._alpha(alpha, False)
        self.buf.append('%s %s %s rg' % (_fmt(r), _fmt(g), _fmt(b)))
    def setStrokeColorRGB(self, r, g, b, alpha=None):
        if alpha is not None:
            self._alpha(alpha, True)
        self.buf.append('%s %s %s RG' % (_fmt(r), _fmt(g), _fmt(b)))
    def setLineWidth(self, w):
        self.buf.append('%s w' % _fmt(w))
    def setLineCap(self, mode):
        self.buf.append('%d J' % mode)
    def setLineJoin(self, mode):
        self.buf.append('%d j' % mode)
    # --- transforms ---
    def transform(self, a, b, c, d, e, f):
        self.buf.append('%s %s %s %s %s %s cm' % tuple(_fmt(v) for v in (a, b, c, d, e, f)))
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
    def _paint(self, fill, stroke):
        self.buf.append('B' if (fill and stroke) else 'f' if fill else 'S' if stroke else 'n')
    def drawPath(self, path, fill=0, stroke=1):
        self.buf.extend(path.ops)
        self._paint(fill, stroke)
    def rect(self, x, y, width, height, fill=0, stroke=1):
        self.buf.append('%s %s %s %s re' % tuple(_fmt(v) for v in (x, y, width, height)))
        self._paint(fill, stroke)
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
        if not text or self._font is None:
            return
        ids = [self._font.ttf.gid(ch) for ch in text]
        self.glyphs.setdefault(self._font.name, set()).update(ids)
        gids = ''.join('%04X' % g for g in ids)
        self.buf.append('BT /%s %s Tf %s Tc 1 0 0 1 %s %s Tm <%s> Tj ET'
                        % (self._font.name, _fmt(self._size), _fmt(charSpace), _fmt(x), _fmt(y), gids))
    # --- metadata ---
    def setTitle(self, v):
        self.info['Title'] = v
    def setAuthor(self, v):
        self.info['Author'] = v
    def setSubject(self, v):
        self.info['Subject'] = v
    def setKeywords(self, v):
        self.info['Keywords'] = v
    # --- output ---
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
        widths, run, start = [], [], None
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
        content = zlib.compress('\n'.join(self.buf).encode('latin-1'), 9)
        stream = out.add(b'<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(content) + content + b'\nendstream')
        fonts = ' '.join('/%s %d 0 R' % (n, self._font_objects(out, n, f)) for n, f in self.used.items())
        gs = ' '.join('/%s << /Type /ExtGState /%s %s >>' % (n, 'CA' if k[1] else 'ca', _fmt(k[0]))
                      for k, n in self.alphas.items())
        res = '<< /Font << %s >> /ExtGState << %s >> >>' % (fonts, gs)
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
