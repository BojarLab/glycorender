"""Minimal TrueType reader: enough for PDF embedding and text measurement."""
import struct
class TTF:
    def __init__(self, path):
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
    def width(self, gid):
        if not self.advances:
            return 0
        a = self.advances[gid] if gid < len(self.advances) else self.advances[-1]
        return a * 1000.0 / self.units_per_em
    def string_width(self, text, size):
        return sum(self.width(self.gid(ch)) for ch in text) * size / 1000.0


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
