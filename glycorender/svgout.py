"""Emit SVG from a pdfwrite display list. Text becomes glyph outlines, so the result carries no font dependency."""
import itertools
import re
from collections import Counter
from .pdfwrite import _fmt, _inv, _mul

_DOC = itertools.count()  # gradient ids must stay unique when several documents are pasted into one figure

_CAP = {0: 'butt', 1: 'round', 2: 'square'}
_JOIN = {0: 'miter', 1: 'round', 2: 'bevel'}


def _num(v):
    """A path coordinate to a hundredth of a unit: 1/7200 inch at the drawing's own size, far below anything visible."""
    s = ('%.2f' % v).rstrip('0').rstrip('.')
    return '0' if s == '-0' else s


def _d(ops):
    # Each command letter runs into its first number, which every SVG reader parses and saves a fifth of the bytes
    return ''.join('Z' if seg[0] == 'h' else {'m': 'M', 'l': 'L', 'c': 'C'}[seg[0]] + ' '.join(_num(v) for v in seg[1:])
                   for seg in ops)


def _hex(rgb):
    return '#%02X%02X%02X' % tuple(max(0, min(255, int(round(v * 255)))) for v in rgb)


def _gradient(op, grads, prefix):
    """Register the op's radial gradient and return the paint reference for it."""
    key = op['fill_grad']
    if key not in grads:
        grads[key] = '%s%d' % (prefix, len(grads))
    return 'url(#%s)' % grads[key]


def _shape(d, op, undo):
    """A path's geometry, with a transform only where its own differs from the one its document's group applies."""
    ctm = tuple(round(v, 9) for v in _mul(op['ctm'], undo))
    if ctm == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0):
        return 'd="%s"' % d
    return 'd="%s" transform="matrix(%s)"' % (d, ' '.join(_fmt(v) for v in ctm))


def _element(d, op, fill, stroke, undo, grads = None, prefix = ''):
    attrs = [_shape(d, op, undo)]
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


def _filter(shadow, name):
    return ('<filter id="%s" x="-30%%" y="-30%%" width="160%%" height="160%%">'
            '<feDropShadow dx="%s" dy="%s" stdDeviation="%s" flood-color="%s" flood-opacity="%s" />'
            '</filter>' % (name, _fmt(shadow['dx']), _fmt(shadow['dy']),
                           _fmt(shadow['blur']), _hex(shadow['color']), _fmt(shadow['alpha'])))


def emit(ops, width, height, shadow = None, sticker = None):
    body, grads, shapes = [], {}, []
    prefix = 'grad%d_' % next(_DOC)
    for op in ops:
        if op['kind'] == 'path':
            if not (op['fill'] or op['stroke']): continue
            shapes.append((op, _d(op['path'])))
        else:
            ttf, pen, segs = op['ttf'], op['x'], []
            gids = [ttf.gid(ch) for ch in op['text']]
            for n, gid in enumerate(gids):
                k = op['size'] / ttf.units_per_em
                for cmd, a in ttf.outline(gid):
                    if cmd == 'M': segs.append('M' + _num(a[0]*k + pen) + ' ' + _num(a[1]*k + op['y']))
                    elif cmd == 'L': segs.append('L' + _num(a[0]*k + pen) + ' ' + _num(a[1]*k + op['y']))
                    elif cmd == 'Q':
                        segs.append('Q' + ' '.join(_num(v) for v in (a[0]*k + pen, a[1]*k + op['y'], a[2]*k + pen,
                                                                     a[3]*k + op['y'])))
                    elif cmd == 'Z': segs.append('Z')
                pen += ttf.width(gid) * op['size'] / 1000.0 + op['char_space']
                if n + 1 < len(gids):
                    pen += ttf.kern(gid, gids[n + 1]) * k
            if segs: shapes.append((op, ''.join(segs)))
    # Shapes mostly share one transform: the drawing's group applies it, so that they need none of their own
    common = Counter(op['ctm'] for op, _ in shapes).most_common(1)[0][0] if shapes else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    undo = _inv(common)
    if sticker:
        layers = [(sticker['width'] + sticker['edge'], sticker['edge_color'])] if sticker['edge'] > 0 else []
        if shadow: body.append(
            '<g filter="url(#%sshadow)">' % prefix)  # one shadow for the whole cut, not one per piece
        for grow, color in layers + [(sticker['width'], sticker['color'])]:
            # Every piece of a layer has the same paint, so the layer's group carries it
            body.append('<g fill="%s" stroke="%s" stroke-linecap="round" stroke-linejoin="round">'
                        % (_hex(color), _hex(color)))
            for op, d in shapes:
                is_path = op['kind'] == 'path'
                filled = not is_path or op['fill']
                line = (op['line_width'] if is_path and op['stroke'] else 0.0) + 2 * grow
                # Chromium's stroker punches holes into glyph quads tighter than the cut is wide, so cut along their chords
                body.append('<path %s%s stroke-width="%s" />' % (
                    _shape(d if is_path else re.sub(r'Q\S+ \S+ ', 'L', d), op, undo),
                    (' fill-rule="evenodd"' if op.get('even_odd') else '') if filled else ' fill="none"', _fmt(line)))
            body.append('</g>')
        if shadow: body.append('</g>')
    for op, d in shapes:
        if op['kind'] == 'path':
            el = _element(d, op, op['fill'], op['stroke'], undo, grads = grads, prefix = prefix)
            if shadow and not sticker and op['fill'] and op['stroke']:
                el = el.replace('<path ', '<path filter="url(#%sshadow)" ' % prefix, 1)
            body.append(el)
        else:
            body.append(_element(d, op, True, False, undo))
    defs, entries = '', []
    if grads:
        for (kind, geo, stops), name in grads.items():
            inner = ''.join('<stop offset="%s" stop-color="%s" stop-opacity="%s" />'
                            % (_fmt(o), _hex(c[:3]), _fmt(c[3])) for o, c in stops)
            if kind == 'radial':
                entries.append(
                    '<radialGradient id="%s" gradientUnits="userSpaceOnUse" cx="%s" cy="%s" r="%s">%s</radialGradient>'
                    % ((name,) + tuple(_fmt(v) for v in geo) + (inner,)))
            else:
                entries.append(
                    '<linearGradient id="%s" gradientUnits="userSpaceOnUse" x1="%s" y1="%s" x2="%s" y2="%s">%s</linearGradient>'
                    % ((name,) + tuple(_fmt(v) for v in geo) + (inner,)))
        defs = ''.join(entries)
    if shadow:
        defs += _filter(shadow, prefix + 'shadow')
    if defs:
        defs = '<defs>%s</defs>\n' % defs
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" viewBox="0 0 %s %s">\n'
            '%s<g transform="matrix(%s)">\n%s\n</g>\n</svg>\n'
            % (_fmt(width), _fmt(height), _fmt(width), _fmt(height), defs,
               ' '.join(_fmt(v) for v in _mul(common, (1.0, 0.0, 0.0, -1.0, 0.0, height))), '\n'.join(body)))