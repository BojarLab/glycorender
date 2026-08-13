"""Scanline rasterizer for the pdfwrite display list: coverage accumulation with numpy, PNG out via zlib."""
import math, struct, zlib
import numpy as np

SS = 4  # vertical subsamples per pixel; horizontal coverage is analytic


def _apply(ctm, x, y):
    a, b, c, d, e, f = ctm
    return (a * x + c * y + e, b * x + d * y + f)


def _mul(m, n):
    """m then n (row-vector convention, matching PDF cm)."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a*A + b*C, a*B + b*D, c*A + d*C, c*B + d*D, e*A + f*C + E, e*B + f*D + F)


def _inv(m):
    a, b, c, d, e, f = m
    det = a * d - b * c
    if abs(det) < 1e-12: return None
    return (d/det, -b/det, -c/det, a/det, (c*f - d*e)/det, (b*e - a*f)/det)


def sample_stops(stops, t):
    """Interpolate normalised gradient stops at t, clamping outside [first, last]."""
    xs = [o for o, _ in stops]
    return [np.interp(t, xs, [c[ch] for _, c in stops]) for ch in range(4)]


def flatten(ops, ctm):
    """Path ops (user space) -> list of device-space point lists, one per subpath."""
    subs, cur, start = [], [], None
    for op in ops:
        k = op[0]
        if k == 'm':
            if len(cur) > 1: subs.append(cur)
            start = _apply(ctm, op[1], op[2])
            cur = [start]
        elif k == 'l':
            cur.append(_apply(ctm, op[1], op[2]))
        elif k == 'c':
            if not cur: continue
            p0 = cur[-1]
            p1, p2, p3 = _apply(ctm, op[1], op[2]), _apply(ctm, op[3], op[4]), _apply(ctm, op[5], op[6])
            span = (abs(p1[0]-p0[0]) + abs(p1[1]-p0[1]) + abs(p2[0]-p1[0]) + abs(p2[1]-p1[1])
                    + abs(p3[0]-p2[0]) + abs(p3[1]-p2[1]))
            n = min(64, max(3, int(span / 3.0) + 3))
            t = np.linspace(0, 1, n + 1)[1:]
            mt = 1 - t
            xs = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
            ys = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
            cur.extend(zip(xs.tolist(), ys.tolist()))
        elif k == 'h':
            if cur and start and cur[-1] != start: cur.append(start)
    if len(cur) > 1: subs.append(cur)
    return subs


def _edges(subs, close):
    """Device-space subpaths -> (n,4) edge array, dropping horizontal edges."""
    out = []
    for pts in subs:
        p = np.asarray(pts, dtype = np.float64)
        if close and (p[0, 0] != p[-1, 0] or p[0, 1] != p[-1, 1]):
            p = np.vstack([p, p[:1]])
        if len(p) < 2: continue
        seg = np.hstack([p[:-1], p[1:]])
        out.append(seg[seg[:, 1] != seg[:, 3]])
    if not out: return np.zeros((0, 4))
    return np.vstack(out)


def coverage(edges, x0, y0, w, h, even_odd = False):
    """Antialiased coverage of `edges` over the pixel box at (x0,y0), size (w,h).

    Each pixel row is sampled at SS sub-scanlines; within a row the crossings are
    splatted at fractional x, so horizontal antialiasing is exact rather than sampled.
    """
    cov = np.zeros((h, w), dtype = np.float64)
    if not len(edges) or w <= 0 or h <= 0: return cov
    ex0, ey0, ex1, ey1 = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    dirs = np.where(ey1 > ey0, 1.0, -1.0)
    slope = (ex1 - ex0) / (ey1 - ey0)
    rows = h * SS
    chunk = max(SS, int(4e6 // max(1, len(edges))) // SS * SS)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        ys = y0 + (np.arange(r0, r1) + 0.5) / SS
        hit = (ey0 <= ys[:, None]) != (ey1 <= ys[:, None])
        ri, ei = np.nonzero(hit)
        if not len(ri): continue
        xs = ex0[ei] + (ys[ri] - ey0[ei]) * slope[ei]
        xs = np.clip(xs - x0, 0.0, float(w))
        ix = np.floor(xs).astype(np.intp)
        frac = xs - ix
        d = dirs[ei]
        acc = np.zeros((r1 - r0, w + 2))
        np.add.at(acc, (ri, ix), d * (1.0 - frac))
        np.add.at(acc, (ri, ix + 1), d * frac)
        wind = np.cumsum(acc, axis = 1)[:, :w]
        if even_odd:
            m = np.abs(wind) % 2.0
            band = np.minimum(m, 2.0 - m)
        else:
            band = np.clip(np.abs(wind), 0.0, 1.0)
        cov[r0 // SS:r1 // SS] += band.reshape(-1, SS, w).sum(axis = 1) / SS
    return np.clip(cov, 0.0, 1.0)


def _norm(poly):
    """Force positive orientation so overlapping stroke pieces union under the nonzero rule."""
    p = np.asarray(poly)
    area = np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))
    return poly if area >= 0 else poly[::-1]


def _arc(cx, cy, r, n = 16):
    t = np.linspace(0, 2 * math.pi, n, endpoint = False)
    return list(zip((cx + r * np.cos(t)).tolist(), (cy + r * np.sin(t)).tolist()))


def _dash(pts, pattern, phase):
    """Split a device-space polyline into dashed runs."""
    pattern = [p for p in pattern if p >= 0]
    if not pattern or not any(pattern): return [pts]
    pattern = [max(p, 1e-6) for p in pattern]
    if len(pattern) % 2: pattern = pattern * 2
    total = sum(pattern)
    pos = phase % total
    idx = 0
    while pos >= pattern[idx]:
        pos -= pattern[idx]
        idx = (idx + 1) % len(pattern)
    left, on = pattern[idx] - pos, idx % 2 == 0
    runs, cur = [], ([pts[0]] if on else [])
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        t = 0.0
        while seg - t > left:
            t += left
            f = t / seg
            p = (ax + (bx - ax) * f, ay + (by - ay) * f)
            if on:
                cur.append(p)
                if len(cur) > 1: runs.append(cur)
                cur = []
            else:
                cur = [p]
            on = not on
            idx = (idx + 1) % len(pattern)
            left = pattern[idx]
        left -= seg - t
        if on: cur.append((bx, by))
    if len(cur) > 1: runs.append(cur)
    return runs


def stroke_polys(subs, width, cap, join, dash = None, phase = 0.0):
    """Device-space polylines -> filled outline polygons (segment quads plus join/cap pieces).

    Every piece is emitted with positive orientation, so the nonzero rule unions the
    overlaps for free and no real outline offsetting is needed.
    """
    hw = max(width, 0.1) / 2.0
    polys = []
    for pts in subs:
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
        if len(pts) < 2:
            if pts and cap == 1: polys.append(_arc(pts[0][0], pts[0][1], hw))
            continue
        for run in (_dash(pts, dash, phase) if dash else [pts]):
            run = [p for i, p in enumerate(run) if i == 0 or p != run[i - 1]]
            if len(run) < 2: continue
            closed = run[0] == run[-1] and len(run) > 2
            for i in range(len(run) - 1):
                (ax, ay), (bx, by) = run[i], run[i + 1]
                dx, dy = bx - ax, by - ay
                ln = math.hypot(dx, dy)
                ux, uy = dx / ln, dy / ln
                nx, ny = -uy * hw, ux * hw
                if cap == 2 and not closed and i in (0, len(run) - 2):
                    if i == 0: ax, ay = ax - ux * hw, ay - uy * hw
                    if i == len(run) - 2: bx, by = bx + ux * hw, by + uy * hw
                polys.append(_norm([(ax + nx, ay + ny), (bx + nx, by + ny), (bx - nx, by - ny), (ax - nx, ay - ny)]))
            joints = list(range(1, len(run) - 1)) + ([0] if closed else [])
            for i in joints:
                (px, py), (cx, cy), (nx2, ny2) = run[-2 if i == 0 else i - 1], run[i], run[i + 1]
                if join == 1:
                    polys.append(_arc(cx, cy, hw))
                    continue
                u1 = ((cx - px), (cy - py))
                u2 = ((nx2 - cx), (ny2 - cy))
                l1, l2 = math.hypot(*u1) or 1.0, math.hypot(*u2) or 1.0
                u1, u2 = (u1[0] / l1, u1[1] / l1), (u2[0] / l2, u2[1] / l2)
                cross = u1[0] * u2[1] - u1[1] * u2[0]
                if abs(cross) < 1e-9: continue
                s = -1.0 if cross > 0 else 1.0
                a = (cx - u1[1] * hw * s, cy + u1[0] * hw * s)
                b = (cx - u2[1] * hw * s, cy + u2[0] * hw * s)
                wedge = [(cx, cy), a, b]
                if join == 0:
                    cosh = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
                    half = (math.pi - math.acos(cosh)) / 2.0
                    if math.sin(half) > 1e-6 and 1.0 / math.sin(half) <= 10.0:  # miter limit
                        mx, my = u1[0] - u2[0], u1[1] - u2[1]
                        ml = math.hypot(mx, my)
                        if ml > 1e-9:
                            d = hw / math.sin(half)
                            wedge = [(cx, cy), a, (cx + mx / ml * d, cy + my / ml * d), b]
                polys.append(_norm(wedge))
            if cap == 1 and not closed:
                polys.append(_arc(run[0][0], run[0][1], hw))
                polys.append(_arc(run[-1][0], run[-1][1], hw))
    return polys


def _clip_box(clips, flip):
    """Clip rects -> one device-space bounding box (axis-aligned approximation)."""
    if not clips: return None
    box = None
    for (x, y, w, h), ctm in clips:
        m = _mul(ctm, flip)
        pts = [_apply(m, x + dx * w, y + dy * h) for dx in (0, 1) for dy in (0, 1)]
        cur = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
        box = cur if box is None else (max(box[0], cur[0]), max(box[1], cur[1]), min(box[2], cur[2]), min(box[3], cur[3]))
    return box


def _glyph_paths(op, ctm):
    """Yield (path ops in font units, ctm) for each glyph of a text op."""
    ttf = op['ttf']
    k = op['size'] / ttf.units_per_em
    pen = op['x']
    gids = [ttf.gid(ch) for ch in op['text']]
    for i, gid in enumerate(gids):
        cmds = ttf.outline(gid)
        if cmds:
            place = _mul((k, 0.0, 0.0, k, pen, op['y']), ctm)
            path, cur, start = [], (0.0, 0.0), (0.0, 0.0)
            for cmd, args in cmds:
                if cmd == 'M':
                    path.append(('m', args[0], args[1]))
                    cur = start = (args[0], args[1])
                elif cmd == 'L':
                    path.append(('l', args[0], args[1]))
                    cur = (args[0], args[1])
                elif cmd == 'Q':
                    x0, y0 = cur
                    cx, cy, x1, y1 = args
                    path.append(('c', x0 + 2.0/3*(cx-x0), y0 + 2.0/3*(cy-y0), x1 + 2.0/3*(cx-x1), y1 + 2.0/3*(cy-y1), x1, y1))
                    cur = (x1, y1)
                elif cmd == 'Z':
                    path.append(('h',))
                    cur = start
            yield path, place
        pen += ttf.width(gid) * op['size'] / 1000.0 + op['char_space']
        if i + 1 < len(gids):
            pen += ttf.kern(gid, gids[i + 1]) * k


class Image:
    """Premultiplied float RGBA accumulation buffer."""
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.buf = np.zeros((h, w, 4), dtype = np.float64)
    def paint(self, edges, rgb, alpha, even_odd = False, box = None):
        if not len(edges) or alpha <= 0: return
        lo = box[:2] if box else (-1e9, -1e9)
        hi = box[2:] if box else (1e9, 1e9)
        x0 = max(0, int(math.floor(max(edges[:, [0, 2]].min(), lo[0]))))
        x1 = min(self.w, int(math.ceil(min(edges[:, [0, 2]].max(), hi[0]))) + 1)
        y0 = max(0, int(math.floor(max(edges[:, [1, 3]].min(), lo[1]))))
        y1 = min(self.h, int(math.ceil(min(edges[:, [1, 3]].max(), hi[1]))) + 1)
        if x1 <= x0 or y1 <= y0: return
        a = coverage(edges, x0, y0, x1 - x0, y1 - y0, even_odd) * alpha
        if not a.any(): return
        sub = self.buf[y0:y1, x0:x1]
        a3 = a[:, :, None]
        sub[:, :, :3] = sub[:, :, :3] * (1 - a3) + np.asarray(rgb, dtype = np.float64) * a3
        sub[:, :, 3] = sub[:, :, 3] * (1 - a) + a
    def paint_grad(self, edges, grad, ctm, alpha, even_odd = False, box = None):
        """Fill `edges` with a radial gradient evaluated per pixel in the op's user space."""
        inv = _inv(ctm)
        if inv is None or not len(edges) or alpha <= 0: return
        lo = box[:2] if box else (-1e9, -1e9)
        hi = box[2:] if box else (1e9, 1e9)
        x0 = max(0, int(math.floor(max(edges[:, [0, 2]].min(), lo[0]))))
        x1 = min(self.w, int(math.ceil(min(edges[:, [0, 2]].max(), hi[0]))) + 1)
        y0 = max(0, int(math.floor(max(edges[:, [1, 3]].min(), lo[1]))))
        y1 = min(self.h, int(math.ceil(min(edges[:, [1, 3]].max(), hi[1]))) + 1)
        if x1 <= x0 or y1 <= y0: return
        cov = coverage(edges, x0, y0, x1 - x0, y1 - y0, even_odd)
        if not cov.any(): return
        kind, geo, stops = grad
        px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        ux = inv[0] * px + inv[2] * py + inv[4]
        uy = inv[1] * px + inv[3] * py + inv[5]
        if kind == 'radial':
            cx, cy, r = geo
            t = np.hypot(ux - cx, uy - cy) / (r if abs(r) > 1e-9 else 1e-9)
        else:
            ax, ay, bx, by = geo
            dx, dy = bx - ax, by - ay
            span = dx * dx + dy * dy
            t = ((ux - ax) * dx + (uy - ay) * dy) / (span if span > 1e-12 else 1e-12)
        cr, cg, cb, ca = sample_stops(stops, t)
        a = cov * ca * alpha
        sub = self.buf[y0:y1, x0:x1]
        a3 = a[:, :, None]
        sub[:, :, :3] = sub[:, :, :3] * (1 - a3) + np.dstack([cr, cg, cb]) * a3
        sub[:, :, 3] = sub[:, :, 3] * (1 - a) + a
    def rgb(self, background = None):
        a = self.buf[:, :, 3:4]
        if background is None:  # keep the alpha channel; the buffer is premultiplied
            out = np.concatenate([np.divide(self.buf[:, :, :3], a, out = np.zeros_like(self.buf[:, :, :3]),
                                            where = a > 1e-6), a], axis = 2)
        else:
            out = self.buf[:, :, :3] + np.asarray(background, dtype = np.float64) * (1 - a)
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def encode_png(arr, texts = ()):
    """uint8 (H,W,3) or (H,W,4) -> PNG bytes, with optional (keyword, value) tEXt chunks."""
    if arr.shape[2] == 4 and arr[:, :, 3].min() == 255:
        arr = arr[:, :, :3]  # nothing is transparent, so don't pay for an alpha channel
    h, w = arr.shape[:2]
    raw = np.hstack([np.zeros((h, 1), dtype = np.uint8), arr.reshape(h, -1)]).tobytes()
    def chunk(tag, body):
        return struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body) & 0xFFFFFFFF)
    out = [b'\x89PNG\r\n\x1a\n', chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6 if arr.shape[2] == 4 else 2, 0, 0, 0))]
    for key, value in texts:
        out.append(chunk(b'tEXt', key.encode('latin-1') + b'\x00' + value.encode('latin-1', 'replace')))
    out.append(chunk(b'IDAT', zlib.compress(raw, 6)))
    out.append(chunk(b'IEND', b''))
    return b''.join(out)


def _box_blur(a, radius):
    """Three box passes approximate a Gaussian closely enough for a shadow."""
    if radius < 1: return a
    k = 2 * int(radius) + 1
    for _ in range(3):
        for axis in (0, 1):
            pad = np.pad(a, ((k // 2, k // 2), (0, 0)) if axis == 0 else ((0, 0), (k // 2, k // 2)))
            c = np.cumsum(pad, axis = axis)
            head = c[k - 1:k] if axis == 0 else c[:, k - 1:k]
            rest = (c[k:] - c[:-k]) if axis == 0 else (c[:, k:] - c[:, :-k])
            a = (np.vstack([head, rest]) if axis == 0 else np.hstack([head, rest])) / k
    return a


def _shadow_mask(ops, w, h, flip, dx, dy):
    """Alpha coverage of every filled-and-stroked shape (i.e., the SNFG symbols), offset."""
    layer = Image(w, h)
    shift = _mul(flip, (1.0, 0.0, 0.0, 1.0, dx, dy))
    for op in ops:
        if op['kind'] != 'path' or not (op['fill'] and op['stroke']):
            continue
        ctm = _mul(op['ctm'], shift)
        if _inv(ctm) is None: continue
        subs = flatten(op['path'], ctm)
        if not subs: continue
        layer.paint(_edges(subs, True), (0.0, 0.0, 0.0), 1.0, op.get('even_odd', False))
        sc = math.sqrt(abs(ctm[0] * ctm[3] - ctm[1] * ctm[2])) or 1.0
        layer.paint(_edges(stroke_polys(subs, op['line_width'] * sc, op['cap'], op['join']), True),
                    (0.0, 0.0, 0.0), 1.0)
    return layer.buf[:, :, 3]


def shadow_alpha(ops, width, height, shadow, dpi = 150.0):
    """Blurred shadow coverage as a uint8 mask, for use as a PDF luminosity soft mask."""
    sc = dpi / 72.0
    w, h = max(1, int(round(width * sc))), max(1, int(round(height * sc)))
    flip = (sc, 0.0, 0.0, -sc, 0.0, float(h))
    mask = _box_blur(_shadow_mask(ops, w, h, flip, shadow['dx'] * sc, shadow['dy'] * sc),
                     max(1, int(round(shadow['blur'] * sc))))
    a = np.clip(mask * shadow['alpha'], 0.0, 1.0)
    return w, h, np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render(ops, width, height, scale_x = 1.0, scale_y = 1.0, background = None, shadow = None):
    """Rasterize a pdfwrite display list; `background` None keeps alpha. PDF y-up is flipped to image y-down here."""
    w = max(1, int(round(width * scale_x)))
    h = max(1, int(round(height * scale_y)))
    flip = (scale_x, 0.0, 0.0, -scale_y, 0.0, float(h))
    img = Image(w, h)
    if shadow:
        sc = (scale_x + scale_y) / 2.0
        mask = _box_blur(_shadow_mask(ops, w, h, flip, shadow['dx'] * sc, shadow['dy'] * sc),
                         max(1, int(round(shadow['blur'] * sc))))
        a = np.clip(mask * shadow['alpha'], 0.0, 1.0)
        img.buf[:, :, :3] = np.asarray(shadow['color'], dtype = np.float64) * a[:, :, None]
        img.buf[:, :, 3] = a
    for op in ops:
        ctm = _mul(op['ctm'], flip)
        if _inv(ctm) is None: continue
        box = _clip_box(op.get('clip'), flip)
        if op['kind'] == 'path':
            subs = flatten(op['path'], ctm)
            if not subs: continue
            if op['fill'] and op.get('fill_grad'):
                img.paint_grad(_edges(subs, True), op['fill_grad'], ctm, op['fill_alpha'], op.get('even_odd', False),
                               box)
            elif op['fill']:
                img.paint(_edges(subs, True), op['fill_rgb'], op['fill_alpha'], op.get('even_odd', False), box)
            if op['stroke']:
                sc = math.sqrt(abs(ctm[0] * ctm[3] - ctm[1] * ctm[2])) or 1.0
                dash = [v * sc for v in op['dash'][0]] if op['dash'] and op['dash'][0] else None
                polys = stroke_polys(subs, op['line_width'] * sc, op['cap'], op['join'], dash,
                                     (op['dash'][1] if op['dash'] else 0.0) * sc)
                img.paint(_edges(polys, True), op['stroke_rgb'], op['stroke_alpha'], False, box)
        else:
            subs = []
            for cmds, gctm in _glyph_paths(op, ctm):
                subs.extend(flatten(cmds, gctm))
            if subs: img.paint(_edges(subs, True), op['fill_rgb'], op['fill_alpha'], False, box)
    return img.rgb(background)