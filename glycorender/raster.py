"""Scanline rasterizer for the pdfwrite display list: coverage accumulation with numpy, PNG out via zlib."""
import math, struct, zlib
import numpy as np

SS = 4  # vertical subsamples per pixel; horizontal coverage is analytic
_UNIT_CIRCLES = {}
# Cubic Bernstein weights for every step count flatten() can pick, so a curve costs four multiply-adds, not a linspace
_BERNSTEIN = {n: ((1 - t)**3, 3*(1 - t)**2*t, 3*(1 - t)*t**2, t**3)
              for n in range(3, 65) for t in [np.linspace(0, 1, n + 1)[1:]]}


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
    a, b, c, d, e, f = ctm
    for op in ops:
        k = op[0]
        if k == 'm':
            if len(cur) > 1: subs.append(cur)
            start = (a * op[1] + c * op[2] + e, b * op[1] + d * op[2] + f)
            cur = [start]
        elif k == 'l':
            cur.append((a * op[1] + c * op[2] + e, b * op[1] + d * op[2] + f))
        elif k == 'c':
            if not cur: continue
            p0 = cur[-1]
            p1 = (a * op[1] + c * op[2] + e, b * op[1] + d * op[2] + f)
            p2 = (a * op[3] + c * op[4] + e, b * op[3] + d * op[4] + f)
            p3 = (a * op[5] + c * op[6] + e, b * op[5] + d * op[6] + f)
            span = (abs(p1[0]-p0[0]) + abs(p1[1]-p0[1]) + abs(p2[0]-p1[0]) + abs(p2[1]-p1[1])
                    + abs(p3[0]-p2[0]) + abs(p3[1]-p2[1]))
            b0, b1, b2, b3 = _BERNSTEIN[min(64, max(3, int(span / 3.0) + 3))]
            xs = b0*p0[0] + b1*p1[0] + b2*p2[0] + b3*p3[0]
            ys = b0*p0[1] + b1*p1[1] + b2*p2[1] + b3*p3[1]
            cur.extend(zip(xs.tolist(), ys.tolist()))
        elif k == 'h':
            if cur and start and cur[-1] != start: cur.append(start)
    if len(cur) > 1: subs.append(cur)
    return subs


def _edges(subs, close):
    """Device-space subpaths -> (n,4) edge array, dropping horizontal edges."""
    # One array for all subpaths, closed where needed, without the segments that would join one subpath to the next;
    # an array per subpath cost more than the edges themselves for the many small quads and wedges of a stroke
    pts, ends = [], []
    for sub in subs:
        pts.extend(sub)
        if close and (sub[0][0] != sub[-1][0] or sub[0][1] != sub[-1][1]):
            pts.append(sub[0])
        ends.append(len(pts) - 1)
    if len(pts) < 2: return np.zeros((0, 4))
    p = np.array(pts, dtype = np.float64)
    seg = np.empty((len(p) - 1, 4))
    seg[:, :2], seg[:, 2:] = p[:-1], p[1:]
    keep = seg[:, 1] != seg[:, 3]
    keep[ends[:-1]] = False
    return seg[keep]


def _loops(*loops):
    """Closed (n,2) point arrays -> (n,4) edge array, dropping horizontal edges."""
    seg = np.vstack([np.hstack([p, np.roll(p, -1, axis = 0)]) for p in loops])
    return seg[seg[:, 1] != seg[:, 3]]


def _convex(sub):
    """A device-space subpath, closed or not, as an (n,2) array turning left at every vertex; None unless convex."""
    p = np.array([q for i, q in enumerate(sub) if i == 0 or q != sub[i - 1]], dtype = np.float64)
    if len(p) > 1 and (p[0] == p[-1]).all(): p = p[:-1]
    if len(p) < 3: return None
    for _ in range(2):
        e = np.roll(p, -1, axis = 0) - p
        nxt = np.roll(e, -1, axis = 0)
        turn = np.arctan2(e[:, 0] * nxt[:, 1] - e[:, 1] * nxt[:, 0], (e * nxt).sum(axis = 1))
        if turn.sum() > 0: break
        p = p[::-1]
    # No turn the other way beyond rounding, and one revolution in all
    return p if turn.min() >= -1e-9 and abs(turn.sum() - 2 * math.pi) <= 1e-6 else None


def _outline(subs, hw, cap, join, filled = False):
    """Edges of the stroke of one convex closed subpath, or of one segment, whose winding is only ever 0 or 1.

    The stroke of a convex polygon is its outer offset less its inner offset, an annulus; `filled` asks for the outer
    offset alone, the union of fill and stroke. A segment, retraced and closed as draw_path leaves a linkage, strokes
    into a capsule. Anything else, open, concave or more than one subpath, is None and goes to stroke_polys.
    """
    if len(subs) != 1: return None
    pts = [q for i, q in enumerate(subs[0]) if i == 0 or q != subs[0][i - 1]]
    closed = len(pts) > 2 and pts[0] == pts[-1]
    if len(pts) == 2 or len(pts) == 3 and closed:
        (ax, ay), (bx, by) = pts[0], pts[1]
        ln = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / ln, (by - ay) / ln
        if (join if closed else cap) == 1:
            # The half discs that the round joins of a retraced segment draw, facet for facet
            n = max(2, int(math.ceil(0.5 * max(16, int(7 * math.sqrt(hw)))))) + 1
            t = math.atan2(uy, ux) - math.pi / 2 + np.arange(n + 1) * (math.pi / n)
            ring = np.column_stack([np.cos(t), np.sin(t)]) * hw
            return _loops(np.vstack([ring + (bx, by), (ax, ay) - ring]))
        ext = hw if cap == 2 and not closed else 0.0
        ax, ay, bx, by = ax - ux * ext, ay - uy * ext, bx + ux * ext, by + uy * ext
        return _loops(np.array([(ax + uy * hw, ay - ux * hw), (bx + uy * hw, by - ux * hw),
                                (bx - uy * hw, by + ux * hw), (ax - uy * hw, ay + ux * hw)]))
    p = _convex(pts) if closed else None
    if p is None: return None
    # Edge i runs from p[i] to p[i + 1]; for a polygon turning left its outward normal is on the right
    e = np.roll(p, -1, axis = 0) - p
    nrm = np.column_stack([e[:, 1], -e[:, 0]]) / np.hypot(e[:, 0], e[:, 1])[:, None]
    prev = np.roll(nrm, 1, axis = 0)
    cos = np.clip((prev * nrm).sum(axis = 1), -1.0, 1.0)
    if join == 1:
        # Each vertex's arc, faceted like the fan stroke_polys gives a round join
        turn = np.maximum(np.arctan2(prev[:, 0] * nrm[:, 1] - prev[:, 1] * nrm[:, 0], cos), 0.0)
        n = np.maximum(2, np.ceil(turn / (2 * math.pi) * max(16, int(7 * math.sqrt(hw)))).astype(np.intp)) + 1
        k = np.arange(n.sum() + len(n)) - np.repeat(np.cumsum(n + 1) - n - 1, n + 1)
        t = np.repeat(np.arctan2(prev[:, 1], prev[:, 0]), n + 1) + k * np.repeat(turn / n, n + 1)
        outer = np.repeat(p, n + 1, axis = 0) + np.column_stack([np.cos(t), np.sin(t)]) * hw
    else:
        # A miter within the limit is the one point where the offset edges meet, anything else the ends of a bevel
        miter = (np.sin((math.pi - np.arccos(cos)) / 2.0) >= 0.1) & (join == 0)
        outer = np.where(miter[:, None], p + (prev + nrm) * (hw / (1.0 + cos))[:, None], p + prev * hw)
        outer = np.insert(outer, np.flatnonzero(~miter) + 1, (p + nrm * hw)[~miter], axis = 0)
    if filled: return _loops(outer)
    inner = p - (prev + nrm) * (hw / (1.0 + cos))[:, None]
    # The inner offset is the polygon eroded, valid only while none of its edges vanishes or turns around
    if ((np.roll(inner, -1, axis = 0) - inner) * e).sum(axis = 1).min() <= 0: return None
    return _loops(outer, inner[::-1])


def coverage(edges, x0, y0, w, h, even_odd = False, simple = False):
    """Antialiased coverage of `edges` over the pixel box at (x0,y0), size (w,h).

    Each pixel row is sampled at SS sub-scanlines; within a row the crossings are
    splatted at fractional x, so horizontal antialiasing is exact rather than sampled.
    `simple` promises a winding number that is only ever 0 or one sign of 1 (a convex
    polygon, an annulus), so the sub-scanlines can be summed before the running sum.
    """
    if not len(edges) or w <= 0 or h <= 0: return np.zeros((h, w), dtype = np.float64)
    ex0, ey0, ex1, ey1 = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    rows = h * SS
    # An edge crosses the sub-scanlines y0 + (r + 0.5) / SS in [min(ey0, ey1), max(ey0, ey1)): the estimate of the first
    # one is off by at most one either way, and testing the two rows from one below it settles it exactly, per edge
    # rather than per crossing; the bounds are integers, so min/max stand in for np.clip, whose wrapper costs more
    def first(v):
        c = np.ceil((v - y0) * SS - 0.5).astype(np.intp) - 1
        return np.minimum(np.maximum(c + (y0 + (c + 0.5) / SS < v) + (y0 + (c + 1.5) / SS < v), 0), rows)
    lo = first(np.minimum(ey0, ey1))
    n = first(np.maximum(ey0, ey1)) - lo
    # Every crossing found is real, so its edge's values come by np.repeat, which copies runs, rather than by an index
    ri = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n - lo, n)
    # Crossings come edge by edge, so within each sub-scanline they are already in edge order, which is all that
    # bincount's per-bin accumulation order depends on (a bin never spans two sub-scanlines); no sort is needed
    xs = np.repeat(ex0, n) + (y0 + (ri + 0.5) / SS - np.repeat(ey0, n)) * np.repeat((ex1 - ex0) / (ey1 - ey0), n)
    xs -= x0
    np.clip(xs, 0.0, float(w), out = xs)
    ix = np.floor(xs).astype(np.intp)
    xs -= ix
    d = np.repeat(np.where(ey1 > ey0, 1.0, -1.0), n)
    hi = d * xs
    if simple:
        # Summed over its sub-scanlines first, a pixel row needs one running sum instead of SS
        ri //= SS
        rows = h
    flat = ri * (w + 2) + ix
    # bincount accumulates in input order exactly like np.add.at, at a fraction of the cost
    chunk = max(SS, int(4e6 // (w + 2)) // SS * SS)
    cov = np.zeros((h, w), dtype = np.float64) if rows > chunk else None
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        if cov is None:
            cf, cd, ch = flat, d, hi
        else:
            sel = (ri >= r0) & (ri < r1)
            if not sel.any(): continue
            cf, cd, ch = flat[sel] - r0 * (w + 2), d[sel], hi[sel]
        acc = np.bincount(np.concatenate([cf, cf + 1]), np.concatenate([cd - ch, ch]),
                          (r1 - r0) * (w + 2)).reshape(r1 - r0, w + 2)
        wind = np.cumsum(acc, axis = 1, out = acc)[:, :w]
        if simple:
            band = np.abs(wind, out = wind)
            band /= SS
            part = np.minimum(band, 1.0, out = band)
        else:
            if even_odd:
                m = np.abs(wind) % 2.0
                band = np.minimum(m, 2.0 - m)
            else:
                band = np.minimum(np.abs(wind, out = wind), 1.0, out = wind)
            # Each row is the mean of SS bands in [0, 1], so it is in [0, 1] itself and needs no clipping
            part = band.reshape(-1, SS, w).sum(axis = 1)
            part /= SS
        if cov is None: return part
        cov[r0 // (1 if simple else SS):r1 // (1 if simple else SS)] = part
    return cov


def _arc(cx, cy, r):
    # Facet count grows with the radius, keeping the sagitta under 0.1 px at any zoom; the facet angles depend only on
    # their count, so their cosines and sines are computed once per count
    n = max(16, int(7 * math.sqrt(r)))
    if n not in _UNIT_CIRCLES:
        t = np.linspace(0, 2 * math.pi, n, endpoint = False)
        _UNIT_CIRCLES[n] = (np.cos(t), np.sin(t))
    cos, sin = _UNIT_CIRCLES[n]
    return list(zip((cx + r * cos).tolist(), (cy + r * sin).tolist()))


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
            first = len(polys)
            for i in range(len(run) - 1):
                (ax, ay), (bx, by) = run[i], run[i + 1]
                dx, dy = bx - ax, by - ay
                ln = math.hypot(dx, dy)
                ux, uy = dx / ln, dy / ln
                nx, ny = -uy * hw, ux * hw
                if cap == 2 and not closed and i in (0, len(run) - 2):
                    if i == 0: ax, ay = ax - ux * hw, ay - uy * hw
                    if i == len(run) - 2: bx, by = bx + ux * hw, by + uy * hw
                # This quad is always clockwise (its area is -2 * ln * hw), so emit it reversed instead of calling _norm
                polys.append([(ax - nx, ay - ny), (bx - nx, by - ny), (bx + nx, by + ny), (ax + nx, ay + ny)])
            joints = list(range(1, len(run) - 1)) + ([0] if closed else [])
            for i in joints:
                (px, py), (cx, cy), (nx2, ny2) = run[-2 if i == 0 else i - 1], run[i], run[i + 1]
                if join == 1:
                    # Only the outer wedge between the two segment ends is uncovered, so a sector does what a whole
                    # disc did; along a flattened curve the turn is a degree or two and the sector a couple of facets
                    a1, a2 = math.atan2(cy - py, cx - px), math.atan2(ny2 - cy, nx2 - cx)
                    turn = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
                    s = math.pi / 2 if turn > 0 else -math.pi / 2
                    n = max(2, int(math.ceil(abs(turn) / (2 * math.pi) * max(16, int(7 * math.sqrt(hw))))) + 1)
                    # A few facets, so math beats numpy's per-call overhead
                    start, stop = a1 - s, a1 - s + turn
                    step = (stop - start) / n
                    arc = [(cx + hw * math.cos(start + k * step), cy + hw * math.sin(start + k * step))
                           for k in range(n)]
                    arc.append((cx + hw * math.cos(stop), cy + hw * math.sin(stop)))
                    # The sector sits against the start of the quad leaving this joint, its last radius on that quad's
                    # end, so it joins the quad as one polygon: a third fewer edges to cross
                    polys[first + i].extend([(cx, cy)] + arc[:-1] if turn > 0 else arc[-2::-1] + [(cx, cy)])
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
                # The wedge's signed area is hw * hw * cross / 2 (the miter tip keeps it convex), so cross is its orientation
                polys.append(wedge if cross > 0 else wedge[::-1])
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
    def __init__(self, w, h):
        """Premultiplied float RGBA accumulation buffer, one plane per channel, so that every blend runs along rows
        instead of across four channels."""
        self.w, self.h = w, h
        self.buf = np.zeros((4, h, w), dtype = np.float32)
    def paint(self, edges, rgb, alpha, even_odd = False, box = None, simple = False):
        if not len(edges) or alpha <= 0: return
        lo = box[:2] if box else (-1e9, -1e9)
        hi = box[2:] if box else (1e9, 1e9)
        xe, ye = edges[:, 0::2], edges[:, 1::2]
        x0 = max(0, int(math.floor(max(xe.min(), lo[0]))))
        x1 = min(self.w, int(math.ceil(min(xe.max(), hi[0]))) + 1)
        y0 = max(0, int(math.floor(max(ye.min(), lo[1]))))
        y1 = min(self.h, int(math.ceil(min(ye.max(), hi[1]))) + 1)
        if x1 <= x0 or y1 <= y0: return
        a = coverage(edges, x0, y0, x1 - x0, y1 - y0, even_odd, simple).astype(np.float32)
        if alpha != 1:
            a *= alpha
        covered = np.count_nonzero(a)
        if not covered: return
        col = np.append(np.asarray(rgb, dtype = np.float32), np.float32(1.0))
        # A mostly covered box (a figure or axes background, a square symbol) blends fastest as one slice
        if 2 * covered > a.size:
            sub = self.buf[:, y0:y1, x0:x1]
            sub *= 1 - a
            sub += col[:, None, None] * a
            return
        # Otherwise blend only the covered pixels; the edge box of a slanted linkage is mostly empty
        ys, xs = np.nonzero(a)
        v = a[ys, xs]
        ys += y0
        xs += x0
        self.buf[:, ys, xs] = self.buf[:, ys, xs] * (1 - v) + col[:, None] * v
    def paint_grad(self, edges, grad, ctm, alpha, even_odd = False, box = None, simple = False):
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
        cov = coverage(edges, x0, y0, x1 - x0, y1 - y0, even_odd, simple)
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
        sub = self.buf[:, y0:y1, x0:x1]
        sub[:3] = sub[:3] * (1 - a) + np.stack([cr, cg, cb]) * a
        sub[3] = sub[3] * (1 - a) + a
    def rgb(self, background = None):
        # Only the inked pixels are computed, gathered by flat index, as a figure is mostly empty page
        flat = self.buf.reshape(4, -1)
        if background is None:  # keep the alpha channel; the buffer is premultiplied
            # Un-premultiply only the inked pixels; everything else is (0, 0, 0, 0)
            out = np.zeros((self.h * self.w, 4), dtype = np.uint8)
            idx = np.flatnonzero(flat[3] > 1e-6)
            px = flat[:, idx]
            v = np.empty((len(idx), 4))
            v[:, :3] = (px[:3] / px[3]).T
            v[:, 3] = px[3]
            v *= 255.0
            v += 0.5
            out[idx] = np.clip(v, 0, 255, out = v).astype(np.uint8)
            return out.reshape(self.h, self.w, 4)
        # Untouched pixels hold zero colour and zero alpha, so they come out as the background colour itself
        bg = np.asarray(background, dtype = np.float64)
        # Filled a whole row at a time, as broadcasting three bytes per pixel is several times slower
        out = np.empty((self.h, self.w * 3), dtype = np.uint8)
        out[...] = np.tile(np.clip(bg * 255.0 + 0.5, 0, 255).astype(np.uint8), self.w)
        out = out.reshape(-1, 3)
        idx = np.flatnonzero(flat[3] != 0)
        px = flat[:, idx]
        v = (1 - px[3]) * bg[:, None]
        v += px[:3]
        v *= 255.0
        v += 0.5
        out[idx] = np.clip(v, 0, 255, out = v).astype(np.uint8).T
        return out.reshape(self.h, self.w, 3)


def encode_png(arr, texts = (), dpi = None):
    """uint8 (H,W,3) or (H,W,4) -> PNG bytes, with optional (keyword, value) tEXt chunks and (x, y) resolution."""
    if arr.shape[2] == 4 and arr[:, :, 3].min() == 255:
        arr = arr[:, :, :3]  # nothing is transparent, so don't pay for an alpha channel
    h, w = arr.shape[:2]
    raw = np.hstack([np.zeros((h, 1), dtype = np.uint8), arr.reshape(h, -1)]).tobytes()
    def chunk(tag, body):
        return struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body) & 0xFFFFFFFF)
    out = [b'\x89PNG\r\n\x1a\n', chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6 if arr.shape[2] == 4 else 2, 0, 0, 0))]
    if dpi:  # pHYs, so Word, PowerPoint, and Inkscape place the PNG at the same physical size as the PDF
        out.append(chunk(b'pHYs', struct.pack('>IIB', int(round(dpi[0] / 0.0254)), int(round(dpi[1] / 0.0254)), 1)))
    for key, value in texts:
        try:
            out.append(chunk(b'tEXt', key.encode('latin-1') + b'\x00' + value.encode('latin-1')))
        except UnicodeEncodeError:  # tEXt is Latin-1 only; iTXt (uncompressed, no language tag) carries UTF-8
            out.append(chunk(b'iTXt', key.encode('latin-1') + b'\x00' * 5 + value.encode('utf-8')))
    out.append(chunk(b'IDAT', zlib.compress(raw, 6)))
    out.append(chunk(b'IEND', b''))
    return b''.join(out)


def _box_blur(a, radius, f = 1):
    """Three box passes approximate a Gaussian closely enough for a shadow; on a grid `f` times coarser, three odd
    widths that come closest to the variance of the full-resolution passes stand in for them."""
    if radius < 1: return a
    k = 2 * int(radius) + 1
    var = (k * k - 1) / 4.0 / (f * f)
    lo = max(1, (int(math.sqrt(4 * var + 1)) - 1) // 2 * 2 + 1)
    ks = min(([lo] * (3 - n) + [lo + 2] * n for n in range(4)),
             key = lambda q: abs(sum(v * v - 1 for v in q) / 12 - var))
    # Only the ink's box, widened by as far as the passes spread it, can come out nonzero
    ys, xs = np.flatnonzero(a.any(axis = 1)), np.flatnonzero(a.any(axis = 0))
    out = np.zeros_like(a)
    if not len(ys): return out
    m = sum(v // 2 for v in ks)
    y0, y1, x0, x1 = max(0, ys[0] - m), ys[-1] + m + 1, max(0, xs[0] - m), xs[-1] + m + 1
    # In float32, each pass divided by its width so that the running sums stay small enough to keep their precision
    b = a[y0:y1, x0:x1].astype(np.float32)
    for k in ks:
        for axis in (0, 1):
            # Along axis 0 of the transposed view for axis 1: a window is a difference of the running sum, and the
            # windows that reach past either end simply stop there, as with zeros beyond
            b = b.T if axis else b
            n, r, c = len(b), k // 2, np.cumsum(b, axis = 0)
            b = np.empty_like(c)
            b[r + 1:n - r] = c[k:] - c[:n - k]
            b[:r + 1] = c[r:k]
            b[n - r:] = c[-1] - c[n - k:n - r - 1]
            b *= 1.0 / k
            b = b.T if axis else b
    out[y0:y1, x0:x1] = b
    return out


def _resample(a, f, w, h, dx, dy):
    """`a`, on a grid `f` times coarser, back on the (h, w) pixel grid and moved by (dx, dy) pixels, interpolated
    linearly, with zeros beyond its edges."""
    out = a
    for axis, n, d in ((1, w, dx), (0, h, dy)):
        # Pixel j = f * q + r samples the coarse grid at q + (r + 0.5 - d) / f - 0.5, so each phase r is two slices
        pad = int(abs(d) / f) + 2
        src = np.pad(out, ((0, 0), (pad, pad)) if axis else ((pad, pad), (0, 0)))
        out = np.empty((src.shape[0], n) if axis else (n, src.shape[1]), dtype = src.dtype)
        for r in range(f):
            u = (r + 0.5 - d) / f - 0.5 + pad
            i, cnt = int(math.floor(u)), len(range(r, n, f))
            t = u - i
            if axis:
                out[:, r::f] = src[:, i:i + cnt] * (1 - t) + src[:, i + 1:i + 1 + cnt] * t
            else:
                out[r::f] = src[i:i + cnt] * (1 - t) + src[i + 1:i + 1 + cnt] * t
    return out


def _shadow(ops, w, h, flip, sc, shadow, sticker, cut = None):
    """The shadow's coverage on the (h, w) grid: the symbols' or, under a die cut, the cut silhouette's, blurred and
    moved into place. A blur this wide leaves nothing that half the resolution could not carry, so from 10 pixels up
    it is drawn and blurred on a quarter of the pixels; `cut` is a full-resolution silhouette that is already drawn."""
    radius = max(1, int(round(shadow['blur'] * sc)))
    f = 2 if radius >= 10 else 1
    if cut is not None:
        base = cut if f == 1 else np.pad(cut, ((0, -h % f), (0, -w % f))).reshape(-(-h // f), f, -(-w // f), f).mean(
            axis = (1, 3))
    else:
        grow = ((sticker['width'] + sticker['edge']) * sc / f) if sticker else 0.0
        base = _ink_mask(ops, -(-w // f), -(-h // f), _mul(flip, (1.0 / f, 0.0, 0.0, 1.0 / f, 0.0, 0.0)), 0.0, 0.0,
                         grow, not sticker)
    return _resample(_box_blur(base, radius, f), f, w, h, shadow['dx'] * sc, shadow['dy'] * sc)


def _ink_mask(ops, w, h, flip, dx = 0.0, dy = 0.0, grow = 0.0, symbols_only = True):
    """Union coverage of the ink, offset by (dx, dy) and grown outward by `grow` device units.

    Coverage is unioned with a max rather than composited, so the seam where two grown shapes
    overlap does not show up as a lighter line in the die-cut silhouette.
    """
    mask = np.zeros((h, w), dtype = np.float64)
    shift = _mul(flip, (1.0, 0.0, 0.0, 1.0, dx, dy))
    def add(edges, even_odd = False, simple = False):
        if not len(edges): return
        x0 = max(0, int(math.floor(edges[:, [0, 2]].min())))
        x1 = min(w, int(math.ceil(edges[:, [0, 2]].max())) + 1)
        y0 = max(0, int(math.floor(edges[:, [1, 3]].min())))
        y1 = min(h, int(math.ceil(edges[:, [1, 3]].max())) + 1)
        if x1 <= x0 or y1 <= y0: return
        sub = mask[y0:y1, x0:x1]
        np.maximum(sub, coverage(edges, x0, y0, x1 - x0, y1 - y0, even_odd, simple), out = sub)
    for op in ops:
        ctm = _mul(op['ctm'], shift)
        if _inv(ctm) is None: continue
        sc = math.sqrt(abs(ctm[0] * ctm[3] - ctm[1] * ctm[2])) or 1.0
        if op['kind'] == 'path':
            if not (op['fill'] and op['stroke'] if symbols_only else op['fill'] or op['stroke']): continue
            subs = flatten(op['path'], ctm)
            if not subs: continue
            width = (op['line_width'] * sc if op['stroke'] else 0.0) + 2.0 * grow
            cap, join = (1, 1) if grow else (op['cap'], op['join'])
            # A filled convex symbol and its stroke make up its outer offset alone, a linkage its capsule
            edges = _outline(subs, max(width, 0.1) / 2.0, cap, join, op['fill']) if width > 0 else None
            if edges is not None:
                add(edges, simple = True)
                continue
            if op['fill']:
                add(_edges(subs, True), op.get('even_odd', False), len(subs) == 1 and _convex(subs[0]) is not None)
            if width > 0: add(_edges(stroke_polys(subs, width, cap, join), True))
        elif not symbols_only:
            subs = []
            for cmds, gctm in _glyph_paths(op, ctm):
                subs.extend(flatten(cmds, gctm))
            if not subs: continue
            add(_edges(subs, True))
            if grow > 0: add(_edges(stroke_polys(subs, 2.0 * grow, 1, 1), True))
    return mask


def shadow_alpha(ops, width, height, shadow, sticker = None, dpi = 150.0):
    """Blurred shadow coverage as 8-bit rows for a PDF soft mask, each a PNG Sub filter byte and the differences to
    its left neighbors: a blur changes little from pixel to pixel, so they pack 30% tighter than the plain values."""
    sc = dpi / 72.0
    w, h = max(1, int(round(width * sc))), max(1, int(round(height * sc)))
    flip = (sc, 0.0, 0.0, -sc, 0.0, float(h))
    a = np.clip(_shadow(ops, w, h, flip, sc, shadow, sticker) * shadow['alpha'], 0.0, 1.0)
    rows = np.ones((h, w + 1), dtype = np.uint8)
    rows[:, 1:] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rows[:, 2:] -= rows[:, 1:-1].copy()
    return w, h, rows.tobytes()


def render(ops, width, height, scale_x = 1.0, scale_y = 1.0, background = None, shadow = None, sticker = None):
    """Rasterize a pdfwrite display list; `background` None keeps alpha. PDF y-up is flipped to image y-down here."""
    w = max(1, int(round(width * scale_x)))
    h = max(1, int(round(height * scale_y)))
    flip = (scale_x, 0.0, 0.0, -scale_y, 0.0, float(h))
    img = Image(w, h)
    sc = (scale_x + scale_y) / 2.0
    if sticker:
        layers = [(sticker['width'] + sticker['edge'], sticker['edge_color'])] if sticker['edge'] > 0 else []
        layers = [(np.clip(_ink_mask(ops, w, h, flip, 0.0, 0.0, grow * sc, False), 0.0, 1.0), color)
                  for grow, color in layers + [(sticker['width'], sticker['color'])]]
    if shadow:
        # A die cut casts the shadow of its outermost silhouette, which is reused rather than drawn again
        a = np.clip(_shadow(ops, w, h, flip, sc, shadow, sticker, layers[0][0] if sticker else None) * shadow['alpha'],
                    0.0, 1.0)
        img.buf[:3] = np.asarray(shadow['color'], dtype = np.float64)[:, None, None] * a
        img.buf[3] = a
    for a, color in layers if sticker else ():
        # In place, plane by plane: the whole page, so temporaries cost more than the arithmetic
        rest = 1 - a
        for c, v in enumerate(tuple(color) + (1.0,)):
            img.buf[c] *= rest
            img.buf[c] += a * v
    for op in ops:
        ctm = _mul(op['ctm'], flip)
        if _inv(ctm) is None: continue
        box = _clip_box(op.get('clip'), flip)
        if op['kind'] == 'path':
            subs = flatten(op['path'], ctm)
            if not subs: continue
            # A single convex outline (a circle, square, diamond or triangle symbol) fills with a winding of 0 or 1
            simple = len(subs) == 1 and _convex(subs[0]) is not None
            if op['fill'] and op.get('fill_grad'):
                img.paint_grad(_edges(subs, True), op['fill_grad'], ctm, op['fill_alpha'], op.get('even_odd', False),
                               box, simple)
            elif op['fill']:
                img.paint(_edges(subs, True), op['fill_rgb'], op['fill_alpha'], op.get('even_odd', False), box, simple)
            if op['stroke']:
                sc = math.sqrt(abs(ctm[0] * ctm[3] - ctm[1] * ctm[2])) or 1.0
                dash = [v * sc for v in op['dash'][0]] if op['dash'] and op['dash'][0] else None
                edges = None if dash else _outline(subs, max(op['line_width'] * sc, 0.1) / 2.0, op['cap'], op['join'])
                fast = edges is not None
                if not fast:
                    edges = _edges(stroke_polys(subs, op['line_width'] * sc, op['cap'], op['join'], dash,
                                                (op['dash'][1] if op['dash'] else 0.0) * sc), True)
                img.paint(edges, op['stroke_rgb'], op['stroke_alpha'], False, box, fast)
        elif op['kind'] == 'image':
            # Nearest-neighbor lookup of each device pixel center in the unit square the image is drawn onto
            inv, corners = _inv(ctm), [_apply(ctm, u, v) for u in (0, 1) for v in (0, 1)]
            lo, hi = (box[:2], box[2:]) if box else ((0, 0), (w, h))
            x0 = max(0, int(math.floor(max(min(p[0] for p in corners), lo[0]))))
            x1 = min(w, int(math.ceil(min(max(p[0] for p in corners), hi[0]))))
            y0 = max(0, int(math.floor(max(min(p[1] for p in corners), lo[1]))))
            y1 = min(h, int(math.ceil(min(max(p[1] for p in corners), hi[1]))))
            if x1 <= x0 or y1 <= y0: continue
            px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            u, v = inv[0] * px + inv[2] * py + inv[4], inv[1] * px + inv[3] * py + inv[5]
            col = np.clip((u * op['w']).astype(np.intp), 0, op['w'] - 1)
            row = np.clip(((1 - v) * op['h']).astype(np.intp), 0, op['h'] - 1)
            a = ((u >= 0) & (u < 1) & (v > 0) & (v <= 1)) * op['fill_alpha']
            if op['alpha'] is not None:
                a = a * np.frombuffer(op['alpha'], np.uint8).reshape(op['h'], op['w'])[row, col] / 255.0
            a3, sub = a[:, :, None], img.buf[y0:y1, x0:x1]
            sub = img.buf[:, y0:y1, x0:x1]
            sub[:3] = sub[:3] * (1 - a) + np.frombuffer(op['rgb'], np.uint8).reshape(
                op['h'], op['w'], 3)[row, col].transpose(2, 0, 1) / 255.0 * a
            sub[3] = sub[3] * (1 - a) + a
        else:
            subs = []
            for cmds, gctm in _glyph_paths(op, ctm):
                subs.extend(flatten(cmds, gctm))
            if subs: img.paint(_edges(subs, True), op['fill_rgb'], op['fill_alpha'], False, box)
    return img.rgb(background)