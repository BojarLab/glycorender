"""General-purpose SVG front-end, scoped to what matplotlib and glycorender's own svgout emit.

Used only by simple_svg_to_pdf/png, i.e. annotate_figure output: a matplotlib figure with
glycan drawings pasted in. Unlike render.py this walks the tree in document order and honors
nested transforms; it is deliberately not a complete SVG implementation.
"""
import math, re
import xml.etree.ElementTree as ET
from . import pdfwrite as canvas
from .pdfwrite import pdfmetrics

_NUM = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?')
_CMD = re.compile(r'([MmLlHhVvCcSsQqTtAaZz])')
_TRANSFORM = re.compile(r'(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)')
_NAMED = {'none': None, 'black': (0, 0, 0), 'white': (1, 1, 1), 'red': (1, 0, 0), 'green': (0, .5, 0),
          'blue': (0, 0, 1), 'yellow': (1, 1, 0), 'cyan': (0, 1, 1), 'magenta': (1, 0, 1),
          'gray': (.502, .502, .502), 'grey': (.502, .502, .502), 'silver': (.753, .753, .753),
          'orange': (1, .647, 0), 'purple': (.5, 0, .5), 'brown': (.647, .165, .165),
          'pink': (1, .753, .796), 'lime': (0, 1, 0), 'navy': (0, 0, .5), 'teal': (0, .5, .5),
          'olive': (.5, .5, 0), 'maroon': (.5, 0, 0), 'darkblue': (0, 0, .545), 'skyblue': (.529, .808, .922)}
_INHERIT = ('fill', 'stroke', 'stroke-width', 'opacity', 'fill-opacity', 'stroke-opacity', 'fill-rule',
            'stroke-linecap', 'stroke-linejoin', 'stroke-dasharray', 'stroke-dashoffset',
            'font-size', 'font-weight', 'text-anchor')
_CAP = {'butt': 0, 'round': 1, 'square': 2}
_JOIN = {'miter': 0, 'round': 1, 'bevel': 2}


def _tag(el):
    return el.tag.rsplit('}', 1)[-1]


def _href(el):
    for key in ('{http://www.w3.org/1999/xlink}href', 'href'):
        if key in el.attrib: return el.attrib[key]
    return None


def _num(v, default=0.0):
    if v is None: return default
    m = _NUM.search(str(v))
    return float(m.group()) if m else default


def _color(text):
    if not text: return None
    text = text.strip().lower()
    if text in _NAMED: return _NAMED[text]
    if text.startswith('#'):
        h = text[1:]
        if len(h) == 3: h = ''.join(c * 2 for c in h)
        if len(h) >= 6:
            return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return None
    if text.startswith('rgb'):
        vals = [float(v) for v in _NUM.findall(text)[:3]]
        return tuple((v / 100.0 if '%' in text else v / 255.0) for v in vals)
    if text.startswith('url('): return None
    return None


def _style(el, parent):
    out = dict(parent)
    for k, v in el.attrib.items():
        if k in _INHERIT: out[k] = v
    for item in el.get('style', '').split(';'):
        if ':' in item:
            k, v = item.split(':', 1)
            if k.strip() in _INHERIT: out[k.strip()] = v.strip()
    return out


def _transform(text):
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for kind, body in _TRANSFORM.findall(text or ''):
        v = [float(x) for x in _NUM.findall(body)]
        if kind == 'matrix' and len(v) == 6: n = tuple(v)
        elif kind == 'translate': n = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif kind == 'scale': n = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif kind == 'rotate':
            r = math.radians(v[0])
            n = (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)
            if len(v) >= 3:
                n = _mul(_mul((1, 0, 0, 1, -v[1], -v[2]), n), (1, 0, 0, 1, v[1], v[2]))
        elif kind == 'skewX': n = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif kind == 'skewY': n = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else: continue
        m = _mul(n, m)
    return m


def _mul(n, m):
    """n applied first, then m."""
    return (n[0]*m[0] + n[1]*m[2], n[0]*m[1] + n[1]*m[3], n[2]*m[0] + n[3]*m[2],
            n[2]*m[1] + n[3]*m[3], n[4]*m[0] + n[5]*m[2] + m[4], n[4]*m[1] + n[5]*m[3] + m[5])


def parse_path(d):
    """Full SVG path grammar, including arcs and scientific notation, as absolute M/L/C/Z ops."""
    out = []
    x = y = sx = sy = 0.0
    px = py = None
    prev = None
    parts = [p for p in _CMD.split(d or '') if p and p.strip()]
    i = 0
    while i < len(parts):
        cmd = parts[i]
        args = [float(v) for v in _NUM.findall(parts[i + 1])] if i + 1 < len(parts) and not _CMD.fullmatch(parts[i + 1]) else []
        i += 2 if args else 1
        rel = cmd.islower()
        k = cmd.upper()
        need = {'M': 2, 'L': 2, 'H': 1, 'V': 1, 'C': 6, 'S': 4, 'Q': 4, 'T': 2, 'A': 7, 'Z': 0}[k]
        chunks = [args[j:j + need] for j in range(0, len(args), need)] if need else [[]]
        for n, a in enumerate(chunks):
            if len(a) < need: break
            if k == 'M':
                x, y = (x + a[0], y + a[1]) if rel else (a[0], a[1])
                if n == 0:
                    out.append(('m', x, y))
                    sx, sy = x, y
                else:
                    out.append(('l', x, y))
                px = py = None
            elif k in 'LHV':
                if k == 'L': x, y = (x + a[0], y + a[1]) if rel else (a[0], a[1])
                elif k == 'H': x = x + a[0] if rel else a[0]
                else: y = y + a[0] if rel else a[0]
                out.append(('l', x, y))
                px = py = None
            elif k in 'CS':
                if k == 'C':
                    c1 = (x + a[0], y + a[1]) if rel else (a[0], a[1])
                    c2, p = ((x + a[2], y + a[3]), (x + a[4], y + a[5])) if rel else ((a[2], a[3]), (a[4], a[5]))
                else:
                    c1 = (2 * x - px, 2 * y - py) if px is not None and prev in 'CS' else (x, y)
                    c2, p = ((x + a[0], y + a[1]), (x + a[2], y + a[3])) if rel else ((a[0], a[1]), (a[2], a[3]))
                out.append(('c', c1[0], c1[1], c2[0], c2[1], p[0], p[1]))
                px, py = c2
                x, y = p
            elif k in 'QT':
                if k == 'Q':
                    q = (x + a[0], y + a[1]) if rel else (a[0], a[1])
                    p = (x + a[2], y + a[3]) if rel else (a[2], a[3])
                else:
                    q = (2 * x - px, 2 * y - py) if px is not None and prev in 'QT' else (x, y)
                    p = (x + a[0], y + a[1]) if rel else (a[0], a[1])
                out.append(('c', x + 2.0/3*(q[0]-x), y + 2.0/3*(q[1]-y),
                            p[0] + 2.0/3*(q[0]-p[0]), p[1] + 2.0/3*(q[1]-p[1]), p[0], p[1]))
                px, py = q
                x, y = p
            elif k == 'A':
                p = (x + a[5], y + a[6]) if rel else (a[5], a[6])
                out.extend(_arc_to_curves(x, y, a[0], a[1], a[2], a[3], a[4], p[0], p[1]))
                x, y = p
                px = py = None
            else:
                out.append(('h',))
                x, y = sx, sy
                px = py = None
        prev = k
    return out


def _arc_to_curves(x1, y1, rx, ry, phi, large, sweep, x2, y2):
    """Endpoint-parameterised elliptical arc -> cubic segments (SVG implementation notes F.6)."""
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2): return [('l', x2, y2)]
    rx, ry = abs(rx), abs(ry)
    r = math.radians(phi)
    cs, sn = math.cos(r), math.sin(r)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p, y1p = cs * dx + sn * dy, -sn * dx + cs * dy
    lam = x1p*x1p / (rx*rx) + y1p*y1p / (ry*ry)
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx*rx * ry*ry - rx*rx * y1p*y1p - ry*ry * x1p*x1p
    den = rx*rx * y1p*y1p + ry*ry * x1p*x1p
    co = math.sqrt(max(0.0, num / den)) * (-1 if bool(large) == bool(sweep) else 1)
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx, cy = cs * cxp - sn * cyp + (x1 + x2) / 2.0, sn * cxp + cs * cyp + (y1 + y2) / 2.0
    t1 = math.atan2((y1p - cyp) / ry, (x1p - cxp) / rx)
    t2 = math.atan2((-y1p - cyp) / ry, (-x1p - cxp) / rx)
    dt = t2 - t1
    if not sweep and dt > 0: dt -= 2 * math.pi
    elif sweep and dt < 0: dt += 2 * math.pi
    n = max(1, int(math.ceil(abs(dt) / (math.pi / 2))))
    out = []
    step = dt / n
    k = 4.0 / 3.0 * math.tan(step / 4.0)
    for i in range(n):
        a, b = t1 + i * step, t1 + (i + 1) * step
        pts = []
        for ang, dk in ((a, k), (b, -k)):
            ex, ey = rx * math.cos(ang), ry * math.sin(ang)
            tx, ty = -rx * math.sin(ang) * dk, ry * math.cos(ang) * dk
            pts.append((cs * (ex + tx) - sn * (ey + ty) + cx, sn * (ex + tx) + cs * (ey + ty) + cy))
        ex, ey = rx * math.cos(b), ry * math.sin(b)
        out.append(('c', pts[0][0], pts[0][1], pts[1][0], pts[1][1],
                    cs * ex - sn * ey + cx, sn * ex + cs * ey + cy))
    return out


class _Renderer:
    def __init__(self, c, font):
        self.c = c
        self.font = font
        self.defs = {}

    def collect(self, el):
        if el.get('id'): self.defs[el.get('id')] = el
        for child in el: self.collect(child)

    def gradient(self, ref, bbox):
        """Resolve a url(#id) paint to (cx, cy, r, stops) in user space, or None."""
        node = self.defs.get(ref[5:-1]) if ref.startswith('url(#') else None
        if node is None or _tag(node) != 'radialGradient': return None
        stops = []
        for stop in node:
            if _tag(stop) != 'stop': continue
            style = _style(stop, {})
            style.update({k: v for k, v in stop.attrib.items() if k.startswith('stop-')})
            for item in stop.get('style', '').split(';'):
                if ':' in item:
                    k, v = item.split(':', 1)
                    style[k.strip()] = v.strip()
            col = _color(style.get('stop-color', 'black')) or (0.0, 0.0, 0.0)
            stops.append((_num(stop.get('offset')), col + (_num(style.get('stop-opacity'), 1.0),)))
        if not stops: return None
        x0, y0, x1, y1 = bbox
        if node.get('gradientUnits') == 'userSpaceOnUse':
            cx, cy, r = _num(node.get('cx')), _num(node.get('cy')), _num(node.get('r'))
        else:
            cx = x0 + (x1 - x0) * _num(node.get('cx'), 0.5)
            cy = y0 + (y1 - y0) * _num(node.get('cy'), 0.5)
            r = _num(node.get('r'), 0.5) * max(x1 - x0, y1 - y0)
        return (cx, cy, r, stops)

    def paint(self, path, style, ctm, clip):
        if not path: return
        raw_fill = str(style.get('fill', 'black')).strip()
        fill = _color(raw_fill)
        stroke = _color(style.get('stroke'))
        grad = None
        if fill is None and raw_fill.startswith('url(#'):
            pts = [(s[-2], s[-1]) for s in path if s[0] != 'h']
            if pts:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                grad = self.gradient(raw_fill, (min(xs), min(ys), max(xs), max(ys)))
            if grad: fill = grad[3][0][1][:3]
        if fill is None and stroke is None: return
        alpha = _num(style.get('opacity'), 1.0)
        self.c.saveState()
        self.c.state['ctm'] = ctm
        self.c.state['clip'] = clip
        p = self.c.beginPath()
        for seg in path:
            {'m': p.moveTo, 'l': p.lineTo, 'c': p.curveTo, 'h': lambda: p.close()}[seg[0]](*seg[1:])
        if fill is not None:
            self.c.setFillColorRGB(*fill, alpha = alpha * _num(style.get('fill-opacity'), 1.0))
            if grad: self.c.setFillGradient(*grad)
        if stroke is not None:
            self.c.setStrokeColorRGB(*stroke, alpha=alpha * _num(style.get('stroke-opacity'), 1.0))
            self.c.setLineWidth(_num(style.get('stroke-width'), 1.0))
            self.c.setLineCap(_CAP.get(str(style.get('stroke-linecap', 'butt')).strip(), 0))
            self.c.setLineJoin(_JOIN.get(str(style.get('stroke-linejoin', 'miter')).strip(), 0))
            dashes = _NUM.findall(style.get('stroke-dasharray', '') or '')
            self.c.setDash([float(v) for v in dashes] or None, _num(style.get('stroke-dashoffset'), 0.0))
        self.c.state['even_odd'] = str(style.get('fill-rule', '')).strip() == 'evenodd'
        self.c.drawPath(p, fill=fill is not None, stroke=stroke is not None)
        self.c.restoreState()

    def walk(self, el, style, ctm, clip, depth=0):
        if depth > 24: return
        tag = _tag(el)
        if tag in ('defs', 'clipPath', 'symbol', 'style', 'metadata', 'title', 'desc'): return
        style = _style(el, style)
        ctm = _mul(_transform(el.get('transform')), ctm)
        clip = self._clip(el, ctm, clip)
        if tag in ('g', 'a', 'svg'):
            if tag == 'svg' and depth:
                ctm = _mul(_viewbox(el), ctm)
            for child in el:
                self.walk(child, style, ctm, clip, depth + 1)
        elif tag == 'use':
            target = self.defs.get((_href(el) or '').lstrip('#'))
            if target is not None:
                shift = _mul((1, 0, 0, 1, _num(el.get('x')), _num(el.get('y'))), ctm)
                self.walk(target, style, shift, clip, depth + 1)
        elif tag == 'path':
            self.paint(parse_path(el.get('d')), style, ctm, clip)
        elif tag == 'rect':
            x, y = _num(el.get('x')), _num(el.get('y'))
            w, h = _num(el.get('width')), _num(el.get('height'))
            self.paint([('m', x, y), ('l', x + w, y), ('l', x + w, y + h), ('l', x, y + h), ('h',)], style, ctm, clip)
        elif tag in ('circle', 'ellipse'):
            cx, cy = _num(el.get('cx')), _num(el.get('cy'))
            r = _num(el.get('r'))
            rx, ry = (r, r) if tag == 'circle' else (_num(el.get('rx')), _num(el.get('ry')))
            self.paint(_ellipse(cx, cy, rx, ry), style, ctm, clip)
        elif tag == 'line':
            self.paint([('m', _num(el.get('x1')), _num(el.get('y1'))), ('l', _num(el.get('x2')), _num(el.get('y2')))],
                       dict(style, fill='none'), ctm, clip)
        elif tag in ('polyline', 'polygon'):
            v = [float(n) for n in _NUM.findall(el.get('points', ''))]
            pts = [('m' if i == 0 else 'l', v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
            if pts: self.paint(pts + ([('h',)] if tag == 'polygon' else []), style, ctm, clip)
        elif tag in ('text', 'tspan'):
            self._text(el, style, ctm, clip, depth)

    def _clip(self, el, ctm, clip):
        ref = (el.get('clip-path') or '').strip()
        if not ref.startswith('url(#'): return clip
        node = self.defs.get(ref[5:-1])
        if node is None: return clip
        inner = _mul(_transform(node.get('transform')), ctm)
        box = None
        for child in node:
            if _tag(child) == 'rect':
                box = (_num(child.get('x')), _num(child.get('y')), _num(child.get('width')), _num(child.get('height')))
            elif _tag(child) == 'path':
                pts = [(s[-2], s[-1]) for s in parse_path(child.get('d')) if s[0] != 'h']
                if pts:
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            if box: break
        if not box: return clip
        return (clip or []) + [(box, inner)]

    def _text(self, el, style, ctm, clip, depth):
        size = _num(style.get('font-size'), 12.0)
        name = self.font[1] if str(style.get('font-weight', '')).strip() in ('bold', '700', '800', '900') else self.font[0]
        text = (el.text or '').strip()
        if text:
            x, y = _num(el.get('x')), _num(el.get('y'))
            anchor = str(style.get('text-anchor', 'start')).strip()
            if anchor in ('middle', 'end'):
                w = pdfmetrics.stringWidth(text, name, size)
                x -= w / 2.0 if anchor == 'middle' else w
            self.c.saveState()
            self.c.state['ctm'] = _mul((1, 0, 0, -1, x, y), ctm)
            self.c.state['clip'] = clip
            fill = _color(style.get('fill', 'black')) or (0, 0, 0)
            self.c.setFillColorRGB(*fill, alpha=_num(style.get('fill-opacity'), 1.0) * _num(style.get('opacity'), 1.0))
            self.c.setFont(name, size)
            self.c.drawString(0, 0, text)
            self.c.restoreState()
        for child in el:
            self.walk(child, style, ctm, clip, depth + 1)


def _ellipse(cx, cy, rx, ry):
    k = 0.5522847498307936
    ox, oy = rx * k, ry * k
    return [('m', cx - rx, cy), ('c', cx - rx, cy - oy, cx - ox, cy - ry, cx, cy - ry),
            ('c', cx + ox, cy - ry, cx + rx, cy - oy, cx + rx, cy),
            ('c', cx + rx, cy + oy, cx + ox, cy + ry, cx, cy + ry),
            ('c', cx - ox, cy + ry, cx - rx, cy + oy, cx - rx, cy), ('h',)]


def _viewbox(root, width=None, height=None):
    vb = [float(v) for v in _NUM.findall(root.get('viewBox', ''))]
    w = width if width is not None else _num(root.get('width'), 0)
    h = height if height is not None else _num(root.get('height'), 0)
    if len(vb) != 4 or vb[2] <= 0 or vb[3] <= 0 or not w or not h:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    sx, sy = w / vb[2], h / vb[3]
    return (sx, 0.0, 0.0, sy, -vb[0] * sx, -vb[1] * sy)


def build(svg_data, target, font=('Comfortaa', 'Comfortaa-Bold')):
    """Render an SVG document into a pdfwrite Canvas in document order."""
    if isinstance(svg_data, bytes): svg_data = svg_data.decode('utf-8')
    root = ET.fromstring(svg_data)
    vb = [float(v) for v in _NUM.findall(root.get('viewBox', ''))]
    width = _num(root.get('width'), vb[2] if len(vb) == 4 else 100.0)
    height = _num(root.get('height'), vb[3] if len(vb) == 4 else 100.0)
    c = canvas.Canvas(target, pagesize=(width, height))
    r = _Renderer(c, font)
    r.collect(root)
    ctm = _mul(_viewbox(root, width, height), (1.0, 0.0, 0.0, -1.0, 0.0, height))
    for child in root:
        r.walk(child, {}, ctm, None, 1)
    return c