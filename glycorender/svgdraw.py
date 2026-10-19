"""Minimal SVG builder for GlycoDraw, in place of drawsvg: the element types glycowork draws with, written out byte for
byte as drawsvg 2.4 writes them, so render.py parses exactly the SVG it always has."""
from xml.sax.saxutils import escape


class Element:
    """An SVG node. Keyword arguments become its attributes (stroke_width is stroke-width, class_ is class,
    xlink__href is xlink:href); an Element as a value becomes a reference to it."""
    TAG_NAME = '_'
    has_content = False
    def __init__(self, children=(), **args):
        self.args = {}
        for k, v in args.items():
            k = k.replace('__', ':').replace('_', '-')
            self.args[k[:-1] if k.endswith('-') else k] = v
        self.children = list(children)
    @property
    def id(self):
        return self.args.get('id')
    def append(self, element):
        self.children.append(element)


class Group(Element):
    TAG_NAME = 'g'
    has_content = True


class Circle(Element):
    TAG_NAME = 'circle'
    def __init__(self, cx, cy, r, **args):
        super().__init__(cx=cx, cy=cy, r=r, **args)


class Rectangle(Element):
    TAG_NAME = 'rect'
    def __init__(self, x, y, width, height, **args):
        super().__init__(x=x, y=y, width=width, height=height, **args)


class Path(Element):
    TAG_NAME = 'path'
    def __init__(self, d='', **args):
        super().__init__(d=d, **args)
    def _command(self, name, *values):
        self.args['d'] += (' ' if self.args['d'] else '') + name + ','.join(map(str, values))
        return self
    def M(self, x, y):
        return self._command('M', x, y)
    def L(self, x, y):
        return self._command('L', x, y)
    def C(self, cx1, cy1, cx2, cy2, ex, ey):
        return self._command('C', cx1, cy1, cx2, cy2, ex, ey)
    def Z(self):
        return self._command('Z')


class Lines(Path):
    def __init__(self, sx, sy, *points, close=False, **args):
        """A polyline through sx, sy, x1, y1, ..., closed into a polygon with close=True."""
        super().__init__(**args)
        self.M(sx, sy)
        for x, y in zip(points[::2], points[1::2]):
            self.L(x, y)
        if close:
            self.Z()


class TSpan(Element):
    TAG_NAME = 'tspan'
    has_content = True
    def __init__(self, text, **args):
        super().__init__(**args)
        self.escaped_text = escape(text)


class TextPath(Element):
    TAG_NAME = 'textPath'
    has_content = True


class Text(Element):
    TAG_NAME = 'text'
    has_content = True
    def __init__(self, text, font_size, x=None, y=None, *, center=False, line_height=1, line_offset=0, path=None,
                 **kwargs):
        """Text at x, y, or along a Path (path=...) in one tspan per line, the first shifted by line_offset em."""
        single = isinstance(text, str) and '\n' not in text and path is None
        lines = (text,) if single else tuple(text.splitlines() if isinstance(text, str) else text)
        if center:
            kwargs.setdefault('text_anchor', 'middle')
            if single:
                kwargs.setdefault('dominant_baseline', 'central')
            else:
                line_offset += 0.5
            line_offset -= line_height * (len(lines) - 1) / 2
        anchor = kwargs.get('text_anchor') if path is not None else None
        super().__init__(x=x, y=y, font_size=font_size, **kwargs)
        self.escaped_text = escape(lines[0]) if single else ''
        if path is None and not single:
            for i, line in enumerate(lines):
                self.append(TSpan(line, x=x, dy=f'{line_offset if i == 0 else line_height}em'))
        elif path is not None:
            offset = {'start': '0', 'middle': '50%', 'end': '100%'}.get(anchor)
            carrier = TextPath(xlink__href=path, **({} if offset is None else {'startOffset': offset}))
            for i, line in enumerate(lines):
                # dx = -1 on end-anchored text is drawsvg's fix for CairoSVG dropping the last character
                if line:
                    carrier.append(TSpan(line, dy=f'{line_offset + i * line_height}em',
                                         **({'dx': -1} if anchor == 'end' else {})))
            self.append(carrier)


class _Stop(Element):
    TAG_NAME = 'stop'


class RadialGradient(Element):
    TAG_NAME = 'radialGradient'
    has_content = True
    def __init__(self, cx, cy, r, gradientUnits='userSpaceOnUse', **args):
        """A radial gradient for fill=...; the drawing that uses it writes it into <defs>."""
        super().__init__(cx=cx, cy=cy, r=r, gradientUnits=gradientUnits, **args)
    def add_stop(self, offset, color, opacity=None, **args):
        stop = _Stop(offset=offset, stop_color=color, stop_opacity=opacity, **args)
        self.append(stop)
        return stop


class Drawing:
    def __init__(self, width, height, origin=(0, 0), id_prefix='d'):
        """The SVG document: width x height user units from origin, element IDs numbered after id_prefix."""
        self.width, self.height = width, height
        self.view_box = tuple(origin) + (width, height)
        self.id_prefix = str(id_prefix)
        self.elements = []
    def append(self, element):
        self.elements.append(element)
    def all_elements(self):
        return list(self.elements)
    def as_svg(self):
        # Three passes, as in drawsvg: anything referenced (a bond carrying its label, a gradient) goes into <defs>
        # under a new ID, a dry run numbers anything drawn twice, and the drawing pass writes a <use> for every
        # element already written, which includes every referenced bond
        out, ids, seen = [], {}, set()
        def ref(el):
            if id(el) not in ids:
                ids[id(el)] = f'{self.id_prefix}{len(ids)}'
        def dup(el):
            was = id(el) in seen
            seen.add(id(el))
            return was
        def defs(el):
            for d in [v for v in el.args.values() if isinstance(v, Element)]:
                if dup(d):
                    continue
                defs(d)
                if d.id is None:
                    ref(d)
                write(d, force=True)
                out.append('\n')
            for child in el.children:
                defs(child)
        def dry(el):
            if dup(el) and el.id is None:
                ref(el)
            for child in el.children:
                dry(child)
        def write(el, force=False):
            if dup(el) and not force:
                out.append(f'<use xlink:href="#{ids.get(id(el), el.id)}" />')
                return
            out.append('<' + el.TAG_NAME)
            for k, v in (dict(el.args, id=ids[id(el)]) if id(el) in ids else el.args).items():
                if isinstance(v, Element):
                    v = ids.get(id(v), v.id)
                    v = None if v is None else f'#{v}' if k == 'xlink:href' else f'url(#{v})'
                if v is not None:
                    out.append(f' {k}="{v}"')
            if not el.has_content and not el.children:
                out.append(' />')
                return
            out.append('>' + getattr(el, 'escaped_text', ''))
            # Text runs its tspans together, every other container puts each child on a line of its own
            sep = '' if isinstance(el, Text) else '\n'
            if el.children:
                out.append(sep)
                for child in el.children:
                    write(child)
                    out.append(sep)
            out.append(f'</{el.TAG_NAME}>')
        view_box = ' '.join(map(str, self.view_box))
        out.append('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
                   'xmlns:xlink="http://www.w3.org/1999/xlink"\n    '
                   f' width="{self.width}" height="{self.height}" viewBox="{view_box}">\n<defs>\n')
        for el in self.elements:
            defs(el)
        out.append('</defs>\n')
        kept = set(seen)
        for el in self.elements:
            dry(el)
        seen = kept
        for el in self.elements:
            write(el)
            out.append('\n')
        out.append('</svg>')
        return ''.join(out)