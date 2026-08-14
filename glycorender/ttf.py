"""Minimal TrueType reader: enough for PDF embedding and text measurement."""
import struct

class TTF:
    def __init__(self, path):
        """Font class."""
        self.path = path
        with open(path, 'rb') as fh:
            self.data = fh.read()
        d = self.data
        num_tables = struct.unpack('>H', d[4:6])[0]
        self.tables = {}
        for i in range(num_tables):
            off = 12 + 16 * i
            tag = d[off:off+4].decode('latin-1')
            t_off, t_len = struct.unpack('>II', d[off+8:off+16])
            self.tables[tag] = (t_off, t_len)
        h = self.tables['head'][0]
        self.units_per_em = struct.unpack('>H', d[h+18:h+20])[0]
        self.bbox = [struct.unpack('>h', d[h+36+2*i:h+38+2*i])[0] for i in range(4)]
        self.mac_style = struct.unpack('>H', d[h+44:h+46])[0]
        hh = self.tables['hhea'][0]
        self.ascent, self.descent = struct.unpack('>hh', d[hh+4:hh+8])
        num_h = struct.unpack('>H', d[hh+34:hh+36])[0]
        hm = self.tables['hmtx'][0]
        self.advances = [struct.unpack('>H', d[hm+4*i:hm+4*i+2])[0] for i in range(num_h)]
        self.num_glyphs = struct.unpack('>H', d[self.tables['maxp'][0]+4:self.tables['maxp'][0]+6])[0]
        self.italic_angle = 0
        if 'post' in self.tables:
            p = self.tables['post'][0]
            self.italic_angle = struct.unpack('>i', d[p+4:p+8])[0] / 65536.0
        self.cap_height = self.ascent
        if 'OS/2' in self.tables:
            o, ol = self.tables['OS/2']
            if ol >= 90:
                self.cap_height = struct.unpack('>h', d[o+88:o+90])[0] or self.ascent
        self.cmap = self._read_cmap()
        self._kern_cache = None
        self._loca_cache = None
        self._outline_cache = {}
    def _read_cmap(self):
        d, (off, _) = self.data, self.tables['cmap']
        n = struct.unpack('>H', d[off+2:off+4])[0]
        best = None
        for i in range(n):
            pid, eid, sub = struct.unpack('>HHI', d[off+4+8*i:off+12+8*i])
            score = {(3, 10): 4, (3, 1): 3, (0, 4): 3, (0, 3): 3, (0, 6): 2}.get((pid, eid), 1)
            if best is None or score > best[0]:
                best = (score, off + sub)
        table = {}
        if best is None:
            return table
        sub = best[1]
        fmt = struct.unpack('>H', d[sub:sub+2])[0]
        if fmt == 4:
            segx2 = struct.unpack('>H', d[sub+6:sub+8])[0]
            seg = segx2 // 2
            ends = struct.unpack('>%dH' % seg, d[sub+14:sub+14+segx2])
            s2 = sub + 16 + segx2
            starts = struct.unpack('>%dH' % seg, d[s2:s2+segx2])
            deltas = struct.unpack('>%dh' % seg, d[s2+segx2:s2+2*segx2])
            ro_at = s2 + 2 * segx2
            ranges = struct.unpack('>%dH' % seg, d[ro_at:ro_at+segx2])
            for i in range(seg):
                for ch in range(starts[i], min(ends[i], 0xFFFF) + 1):
                    if ranges[i] == 0:
                        g = (ch + deltas[i]) & 0xFFFF
                    else:
                        gi = ro_at + 2 * i + ranges[i] + 2 * (ch - starts[i])
                        if gi + 2 > len(d):
                            continue
                        g = struct.unpack('>H', d[gi:gi+2])[0]
                        if g:
                            g = (g + deltas[i]) & 0xFFFF
                    if g:
                        table[ch] = g
        elif fmt == 12:
            ngroups = struct.unpack('>I', d[sub+12:sub+16])[0]
            for i in range(ngroups):
                s, e, gs = struct.unpack('>III', d[sub+16+12*i:sub+28+12*i])
                for ch in range(s, min(e, s + 0x10000) + 1):
                    table[ch] = gs + (ch - s)
        return table
    def gid(self, ch):
        return self.cmap.get(ord(ch), 0)
    def _loca(self):
        if self._loca_cache is None:
            d, n = self.data, self.num_glyphs
            l_off = self.tables['loca'][0]
            if struct.unpack('>h', d[self.tables['head'][0]+50:self.tables['head'][0]+52])[0]:
                self._loca_cache = struct.unpack('>%dI' % (n+1), d[l_off:l_off+4*(n+1)])
            else:
                self._loca_cache = [2*v for v in struct.unpack('>%dH' % (n+1), d[l_off:l_off+2*(n+1)])]
        return self._loca_cache
    def outline(self, gid, depth=0):
        """Glyph outline in font units as M/L/Q/Z commands (y up)."""
        if gid in self._outline_cache:
            return self._outline_cache[gid]
        cmds = []
        if 'glyf' not in self.tables or gid >= self.num_glyphs or depth > 4:
            return cmds
        loca, g_off, d = self._loca(), self.tables['glyf'][0], self.data
        start, end = g_off + loca[gid], g_off + loca[gid+1]
        if end - start >= 10:
            if struct.unpack('>h', d[start:start+2])[0] >= 0:
                for contour in _simple_outline(d, start):
                    cmds.extend(_contour_to_cmds(contour))
            else:
                pos = start + 10
                while True:
                    flags, comp = struct.unpack('>HH', d[pos:pos+4])
                    pos += 4
                    if flags & 1:
                        a1, a2 = struct.unpack('>hh', d[pos:pos+4])
                        pos += 4
                    else:
                        a1, a2 = struct.unpack('>bb', d[pos:pos+2])
                        pos += 2
                    xx = yy = 1.0
                    xy = yx = 0.0
                    if flags & 8:
                        xx = yy = struct.unpack('>h', d[pos:pos+2])[0] / 16384.0
                        pos += 2
                    elif flags & 0x40:
                        xx, yy = [v / 16384.0 for v in struct.unpack('>hh', d[pos:pos+4])]
                        pos += 4
                    elif flags & 0x80:
                        xx, yx, xy, yy = [v / 16384.0 for v in struct.unpack('>hhhh', d[pos:pos+8])]
                        pos += 8
                    dx, dy = (a1, a2) if flags & 2 else (0, 0)
                    for cmd, args in self.outline(comp, depth + 1):
                        out = []
                        for i in range(0, len(args), 2):
                            x, y = args[i], args[i+1]
                            out += [xx * x + xy * y + dx, yx * x + yy * y + dy]
                        cmds.append((cmd, tuple(out)))
                    if not flags & 0x20:
                        break
        self._outline_cache[gid] = cmds
        return cmds
    def width(self, gid):
        if not self.advances:
            return 0
        a = self.advances[gid] if gid < len(self.advances) else self.advances[-1]
        return a * 1000.0 / self.units_per_em
    def kern(self, left, right):
        """Horizontal kerning adjustment between two glyphs, in font units."""
        if self._kern_cache is None:
            try:
                self._kern_cache = _read_kerning(self)
            except Exception:
                self._kern_cache = {}  # malformed GPOS: set text unkerned rather than fail
        return self._kern_cache.get((left, right), 0)
    def string_width(self, text, size):
        gids = [self.gid(ch) for ch in text]
        total = sum(self.width(g) for g in gids)
        total += sum(self.kern(a, b) for a, b in zip(gids, gids[1:])) * 1000.0 / self.units_per_em
        return total * size / 1000.0


def subset(font, gids):
    """Rebuild a TrueType file keeping only `gids` (glyph ids stay stable, so Identity-H needs no remapping)."""
    d, tables = font.data, font.tables
    if 'glyf' not in tables or 'loca' not in tables:
        return d  # CFF-flavoured or otherwise unsupported: ship it whole
    head_off = tables['head'][0]
    long_loca = struct.unpack('>h', d[head_off+50:head_off+52])[0]
    g_off, _ = tables['glyf']
    l_off, _ = tables['loca']
    n = font.num_glyphs
    if long_loca:
        loca = list(struct.unpack('>%dI' % (n+1), d[l_off:l_off+4*(n+1)]))
    else:
        loca = [2*v for v in struct.unpack('>%dH' % (n+1), d[l_off:l_off+2*(n+1)])]
    keep, pending = set(), list(gids) + [0]
    while pending:  # composite glyphs pull in their components
        gid = pending.pop()
        if gid in keep or gid >= n:
            continue
        keep.add(gid)
        start, end = g_off + loca[gid], g_off + loca[gid+1]
        if end - start < 10 or struct.unpack('>h', d[start:start+2])[0] >= 0:
            continue
        pos = start + 10
        while True:
            flags, comp = struct.unpack('>HH', d[pos:pos+4])
            pending.append(comp)
            pos += 8 if flags & 1 else 6
            if flags & 8: pos += 2
            elif flags & 0x40: pos += 4
            elif flags & 0x80: pos += 8
            if not flags & 0x20:
                break
    glyf, new_loca = bytearray(), [0]
    for gid in range(n):
        if gid in keep:
            glyf += d[g_off+loca[gid]:g_off+loca[gid+1]]
        while len(glyf) % 4:
            glyf += b'\x00'
        new_loca.append(len(glyf))
    out = {'glyf': bytes(glyf), 'loca': struct.pack('>%dI' % (n+1), *new_loca)}
    for tag in ('head', 'hhea', 'maxp', 'hmtx', 'cvt ', 'fpgm', 'prep'):
        if tag in tables:
            o, ln = tables[tag]
            out[tag] = d[o:o+ln]
    out['head'] = out['head'][:50] + struct.pack('>h', 1) + out['head'][52:]  # loca is now long format
    tags = sorted(out)
    header = struct.pack('>IHHHH', 0x00010000, len(tags), 0, 0, 0)
    offset = len(header) + 16 * len(tags)
    directory, body = b'', b''
    for tag in tags:
        blob = out[tag] + b'\x00' * (-len(out[tag]) % 4)
        directory += tag.encode('latin-1') + struct.pack('>III', 0, offset + len(body), len(out[tag]))
        body += blob
    return header + directory + body


def _simple_outline(d, start):
    ncont = struct.unpack('>h', d[start:start+2])[0]
    ends = struct.unpack('>%dH' % ncont, d[start+10:start+10+2*ncont])
    npts = ends[-1] + 1 if ncont else 0
    pos = start + 10 + 2 * ncont
    pos += 2 + struct.unpack('>H', d[pos:pos+2])[0]
    flags = []
    while len(flags) < npts:
        f = d[pos]
        pos += 1
        flags.append(f)
        if f & 8:
            rep = d[pos]
            pos += 1
            flags.extend([f] * rep)
    flags = flags[:npts]
    xs, v = [], 0
    for f in flags:
        if f & 2:
            dx = d[pos]
            pos += 1
            v += dx if f & 16 else -dx
        elif not f & 16:
            v += struct.unpack('>h', d[pos:pos+2])[0]
            pos += 2
        xs.append(v)
    ys, v = [], 0
    for f in flags:
        if f & 4:
            dy = d[pos]
            pos += 1
            v += dy if f & 32 else -dy
        elif not f & 32:
            v += struct.unpack('>h', d[pos:pos+2])[0]
            pos += 2
        ys.append(v)
    contours, first = [], 0
    for end in ends:
        pts = [(xs[i], ys[i], bool(flags[i] & 1)) for i in range(first, end + 1)]
        first = end + 1
        if pts:
            contours.append(pts)
    return contours


def _contour_to_cmds(pts):
    """TrueType quadratic contour (with implied on-curve midpoints) -> M/L/Q/Z commands."""
    if not pts:
        return []
    if not pts[0][2]:  # contour starts off-curve: rotate to an on-curve point or synthesise one
        on = next((i for i, p in enumerate(pts) if p[2]), None)
        if on is None:
            mx, my = (pts[0][0] + pts[-1][0]) / 2.0, (pts[0][1] + pts[-1][1]) / 2.0
            pts = [(mx, my, True)] + pts
        else:
            pts = pts[on:] + pts[:on]
    cmds = [('M', (float(pts[0][0]), float(pts[0][1])))]
    i, n = 1, len(pts)
    while i <= n:
        x, y, on = pts[i % n]
        if on:
            cmds.append(('L', (float(x), float(y))))
            i += 1
            continue
        nx, ny, non = pts[(i + 1) % n]
        if not non:
            nx, ny = (x + nx) / 2.0, (y + ny) / 2.0
            cmds.append(('Q', (float(x), float(y), nx, ny)))
            i += 1
        else:
            cmds.append(('Q', (float(x), float(y), float(nx), float(ny))))
            i += 2
    cmds.append(('Z', ()))
    return cmds

def _coverage(d, off):
    """Coverage table -> {glyph: index}."""
    fmt = struct.unpack('>H', d[off:off + 2])[0]
    if fmt == 1:
        n = struct.unpack('>H', d[off + 2:off + 4])[0]
        return {g: i for i, g in enumerate(struct.unpack('>%dH' % n, d[off + 4:off + 4 + 2 * n]))}
    n = struct.unpack('>H', d[off + 2:off + 4])[0]
    out = {}
    for i in range(n):
        start, end, base = struct.unpack('>HHH', d[off + 4 + 6 * i:off + 10 + 6 * i])
        for g in range(start, end + 1):
            out[g] = base + g - start
    return out

def _class_def(d, off):
    """ClassDef table -> {glyph: class}; unlisted glyphs are class 0."""
    fmt = struct.unpack('>H', d[off:off + 2])[0]
    out = {}
    if fmt == 1:
        start, n = struct.unpack('>HH', d[off + 2:off + 6])
        for i, cls in enumerate(struct.unpack('>%dH' % n, d[off + 6:off + 6 + 2 * n])):
            out[start + i] = cls
    else:
        n = struct.unpack('>H', d[off + 2:off + 4])[0]
        for i in range(n):
            lo, hi, cls = struct.unpack('>HHH', d[off + 4 + 6 * i:off + 10 + 6 * i])
            for g in range(lo, hi + 1):
                out[g] = cls
    return out

def _pair_subtable(d, st, pairs):
    fmt, cov_off, vf1, vf2 = struct.unpack('>HHHH', d[st:st + 8])
    if not vf1 & 0x0004:  # no x-advance on the first glyph: nothing that moves text along
        return
    size1 = 2 * bin(vf1).count('1')
    size2 = 2 * bin(vf2).count('1')
    skip = 2 * bin(vf1 & 0x0003).count('1')  # x/y placement precede x-advance in a ValueRecord
    coverage = _coverage(d, st + cov_off)
    if fmt == 1:
        n = struct.unpack('>H', d[st + 8:st + 10])[0]
        first = {i: g for g, i in coverage.items()}
        for i in range(n):
            ps = st + struct.unpack('>H', d[st + 10 + 2 * i:st + 12 + 2 * i])[0]
            cnt = struct.unpack('>H', d[ps:ps + 2])[0]
            for k in range(cnt):
                rec = ps + 2 + k * (2 + size1 + size2)
                second = struct.unpack('>H', d[rec:rec + 2])[0]
                adj = struct.unpack('>h', d[rec + 2 + skip:rec + 4 + skip])[0]
                if adj:
                    pairs[(first[i], second)] = adj
    elif fmt == 2:
        cd1_off, cd2_off, n1, n2 = struct.unpack('>HHHH', d[st + 8:st + 16])
        cd1, cd2 = _class_def(d, st + cd1_off), _class_def(d, st + cd2_off)
        by_class1, by_class2 = {}, {}
        for g in coverage:
            by_class1.setdefault(cd1.get(g, 0), []).append(g)
        for g, cls in cd2.items():
            by_class2.setdefault(cls, []).append(g)
        rec_size = size1 + size2
        for c1 in range(n1):
            if c1 not in by_class1:
                continue
            for c2 in range(n2):
                if c2 not in by_class2:
                    continue
                rec = st + 16 + (c1 * n2 + c2) * rec_size
                adj = struct.unpack('>h', d[rec + skip:rec + 2 + skip])[0]
                if not adj:
                    continue
                if len(pairs) > 200000:  # pathological font: stop rather than blow up memory
                    return
                for g1 in by_class1[c1]:
                    for g2 in by_class2[c2]:
                        pairs[(g1, g2)] = adj

def _legacy_kern(font):
    """Pairs from the old TrueType 'kern' table, which pre-OpenType fonts use instead of GPOS."""
    d = font.data
    off, _ = font.tables['kern']
    n_tables = struct.unpack('>H', d[off + 2:off + 4])[0]
    pos = off + 4
    pairs = {}
    for _ in range(n_tables):
        length, coverage = struct.unpack('>HH', d[pos + 2:pos + 6])
        if coverage & 0x0001 and not coverage & 0x0006 and (
                coverage >> 8) == 0:  # horizontal, real kerning, format 0
            n = struct.unpack('>H', d[pos + 6:pos + 8])[0]
            for i in range(n):
                rec = pos + 14 + 6 * i
                left, right, value = struct.unpack('>HHh', d[rec:rec + 6])
                if value:
                    pairs[(left, right)] = value
        pos += length or 6
    return pairs

def _read_kerning(font):
    """Pair adjustments from the GPOS 'kern' feature -> {(left, right): x-advance in font units}."""
    d = font.data
    if 'GPOS' not in font.tables:
        return _legacy_kern(font) if 'kern' in font.tables else {}
    base = font.tables['GPOS'][0]
    _script_off, feature_off, lookup_off = [base + v for v in struct.unpack('>HHH', d[base + 4:base + 10])]
    wanted = set()
    n_feat = struct.unpack('>H', d[feature_off:feature_off + 2])[0]
    for i in range(n_feat):
        rec = feature_off + 2 + 6 * i
        if d[rec:rec + 4] != b'kern':
            continue
        f = feature_off + struct.unpack('>H', d[rec + 4:rec + 6])[0]
        n_lu = struct.unpack('>H', d[f + 2:f + 4])[0]
        wanted.update(struct.unpack('>%dH' % n_lu, d[f + 4:f + 4 + 2 * n_lu]))
    if not wanted:
        return _legacy_kern(font) if 'kern' in font.tables else {}
    n_lookups = struct.unpack('>H', d[lookup_off:lookup_off + 2])[0]
    pairs = {}
    for idx in sorted(wanted):
        if idx >= n_lookups:
            continue
        lo = lookup_off + struct.unpack('>H', d[lookup_off + 2 + 2 * idx:lookup_off + 4 + 2 * idx])[0]
        kind, _flag, n_sub = struct.unpack('>HHH', d[lo:lo + 6])
        for j in range(n_sub):
            st = lo + struct.unpack('>H', d[lo + 6 + 2 * j:lo + 8 + 2 * j])[0]
            if kind == 9:  # extension lookup: hop to the real subtable
                real, delta = struct.unpack('>HI', d[st + 2:st + 8])
                if real != 2:
                    continue
                st += delta
            elif kind != 2:
                continue
            _pair_subtable(d, st, pairs)
    if not pairs and 'kern' in font.tables:  # GPOS present but no usable kern feature
        return _legacy_kern(font)
    return pairs