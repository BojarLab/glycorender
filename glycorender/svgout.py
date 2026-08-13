"""Emit SVG from a pdfwrite display list. Text becomes glyph outlines, so the result carries no font dependency."""
import itertools
from .pdfwrite import _fmt

_DOC = itertools.count()  # gradient ids must stay unique when several documents are pasted into one figure

_CAP = {0: 'butt', 1: 'round', 2: 'square'}
_JOIN = {0: 'miter', 1: 'round', 2: 'bevel'}


def _d(ops):
    out = []
    for seg in ops:
        if seg[0] == 'h':
            out.append('Z')
        else:
            out.append({'m': 'M', 'l': 'L', 'c': 'C'}[seg[0]] + ' ' + ' '.join(_fmt(v) for v in seg[1:]))
    return ' '.join(out)


def _hex(rgb):
    return '#%02X%02X%02X' % tuple(max(0, min(255, int(round(v * 255)))) for v in rgb)


def _gradient(op, grads, prefix):
    """Register the op's radial gradient and return the paint reference for it."""
    key = op['fill_grad']
    if key not in grads:
        grads[key] = '%s%d' % (prefix, len(grads))
    return 'url(#%s)' % grads[key]


def _element(d, op, fill, stroke, grads = None, prefix = ''):
    attrs = ['d="%s"' % d]
    ctm = op['ctm']
    if ctm != (1.0, 0.0, 0.0, 1.0, 0.0, 0.0):
        attrs.append('transform="matrix(%s)"' % ' '.join(_fmt(v) for v in ctm))
    if fill:
        attrs.append('fill="%s"' % (_gradient(op, grads, prefix) if op.get('fill_grad') and grads is not None
                                    else _hex(op['fill_rgb'])))
        if op['fill_alpha'] < 1: attrs.append('fill-opacity="%s"' % _fmt(op['fill_alpha']))
        if op.get('even_odd'): attrs.append('fill-rule="evenodd"')
    else:
        attrs.append('fill="none"')
    if stroke:
        attrs.append('stroke="%s"' % _hex(op['stroke_rgb']))
        attrs.append('stroke-width="%s"' % _fmt(op['line_width']))
        if op['stroke_alpha'] < 1: attrs.append('stroke-opacity="%s"' % _fmt(op['stroke_alpha']))
        if op['cap']: attrs.append('stroke-linecap="%s"' % _CAP[op['cap']])
        if op['join']: attrs.append('stroke-linejoin="%s"' % _JOIN[op['join']])
        if op['dash'] and op['dash'][0]:
            attrs.append('stroke-dasharray="%s"' % ','.join(_fmt(v) for v in op['dash'][0]))
            if op['dash'][1]: attrs.append('stroke-dashoffset="%s"' % _fmt(op['dash'][1]))
    return '<path ' + ' '.join(attrs) + ' />'


def emit(ops, width, height):
    body, grads = [], {}
    prefix = 'grad%d_' % next(_DOC)
    for op in ops:
        if op['kind'] == 'path':
            if not (op['fill'] or op['stroke']): continue
            body.append(_element(_d(op['path']), op, op['fill'], op['stroke'], grads, prefix))
        else:
            ttf, pen, segs = op['ttf'], op['x'], []
            for ch in op['text']:
                gid = ttf.gid(ch)
                k = op['size'] / ttf.units_per_em
                for cmd, a in ttf.outline(gid):
                    if cmd == 'M': segs.append('M ' + _fmt(a[0]*k + pen) + ' ' + _fmt(a[1]*k + op['y']))
                    elif cmd == 'L': segs.append('L ' + _fmt(a[0]*k + pen) + ' ' + _fmt(a[1]*k + op['y']))
                    elif cmd == 'Q': segs.append('Q ' + ' '.join(_fmt(v) for v in (a[0]*k+pen, a[1]*k+op['y'], a[2]*k+pen, a[3]*k+op['y'])))
                    elif cmd == 'Z': segs.append('Z')
                pen += ttf.width(gid) * op['size'] / 1000.0 + op['char_space']
            if segs: body.append(_element(' '.join(segs), op, True, False))
    defs = ''
    if grads:
        entries = []
        for (cx, cy, r, stops), name in grads.items():
            inner = ''.join('<stop offset="%s" stop-color="%s" stop-opacity="%s" />'
                            % (_fmt(o), _hex(c[:3]), _fmt(c[3])) for o, c in stops)
            entries.append(
                '<radialGradient id="%s" gradientUnits="userSpaceOnUse" cx="%s" cy="%s" r="%s">%s</radialGradient>'
                % (name, _fmt(cx), _fmt(cy), _fmt(r), inner))
        defs = '<defs>%s</defs>\n' % ''.join(entries)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" viewBox="0 0 %s %s">\n'
            '%s<g transform="matrix(1 0 0 -1 0 %s)">\n%s\n</g>\n</svg>\n'
            % (_fmt(width), _fmt(height), _fmt(width), _fmt(height), defs, _fmt(height), '\n'.join(body)))