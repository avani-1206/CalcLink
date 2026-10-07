"""
minih5.py — a tiny read-only HDF5 reader (pure Python + NumPy).

Why? Keras saves weights as HDF5 (.h5). The usual reader, h5py, and the usual
converter, TensorFlow + tf2onnx, are big installs (and TensorFlow doesn't
support Python 3.13 yet). Keras files written by h5py with default settings use
the "classic" HDF5 layout, which is simple enough to read in ~150 lines:

  superblock ─► root group ─► B-tree ─► symbol nodes ─► (name, object header)
  object header messages: dataspace (shape) · datatype (float32…) · layout (where the bytes are)

Supported: superblock v0/v1, object header v1 (+continuations), symbol-table
groups, contiguous or compact float/int datasets. That covers Keras 3
`model.weights.h5`. Anything else raises a clear error.
"""
import struct
import numpy as np


class H5Error(Exception):
    pass


class MiniH5:
    def __init__(self, data: bytes):
        self.b = data
        if data[:8] != b'\x89HDF\r\n\x1a\n':
            raise H5Error('not an HDF5 file')
        ver = data[8]
        if ver not in (0, 1):
            raise H5Error(f'superblock v{ver} not supported (file saved with libver="latest"?)')
        self.so, self.sl = data[13], data[14]          # sizes of offsets / lengths
        p = 24 + (4 if ver == 1 else 0)
        p += 4 * self.so                                # base, free-space, EOF, driver addresses
        self.root = self._sym_entry(p)

    # ── primitive readers ───────────────────────────────────────────────
    def _u(self, p, n):
        return int.from_bytes(self.b[p:p + n], 'little')

    def _sym_entry(self, p):
        name_off = self._u(p, self.so); p += self.so
        header = self._u(p, self.so); p += self.so
        cache = self._u(p, 4); p += 8
        btree = heap = None
        if cache == 1:
            btree = self._u(p, self.so); heap = self._u(p + self.so, self.so)
        return {'name_off': name_off, 'header': header, 'btree': btree, 'heap': heap}

    def _heap_string(self, heap, off):
        if self.b[heap:heap + 4] != b'HEAP':
            raise H5Error('bad local heap')
        data = self._u(heap + 8 + 2 * self.sl, self.so)
        s = data + off
        return self.b[s:self.b.index(b'\0', s)].decode()

    def _messages(self, addr):
        b = self.b
        if b[addr:addr + 4] == b'OHDR':
            raise H5Error('object header v2 not supported')
        if b[addr] != 1:
            raise H5Error(f'unknown object header version {b[addr]}')
        n = self._u(addr + 2, 2)
        size = self._u(addr + 8, 4)
        blocks = [(addr + 16, size)]
        out = []
        while blocks and len(out) < n:
            p, length = blocks.pop(0)
            end = p + length
            while p + 8 <= end and len(out) < n:
                mtype, msize = self._u(p, 2), self._u(p + 2, 2)
                body = p + 8
                if mtype == 0x10:                       # continuation → more messages elsewhere
                    blocks.append((self._u(body, self.so), self._u(body + self.so, self.sl)))
                out.append((mtype, body, msize))
                p = body + msize
        return out

    # ── groups ─────────────────────────────────────────────────────────
    def _children(self, header_addr, btree=None, heap=None):
        if btree is None:
            for mtype, body, _ in self._messages(header_addr):
                if mtype == 0x11:                       # symbol table message
                    btree, heap = self._u(body, self.so), self._u(body + self.so, self.so)
                elif mtype in (0x02, 0x06):
                    raise H5Error('new-style (link) groups not supported')
        if btree is None:
            return {}
        kids = {}
        self._walk_btree(btree, heap, kids)
        return kids

    def _walk_btree(self, addr, heap, kids):
        b = self.b
        if b[addr:addr + 4] != b'TREE':
            raise H5Error('bad B-tree node')
        level, used = b[addr + 5], self._u(addr + 6, 2)
        p = addr + 8 + 2 * self.so
        for i in range(used):
            p += self.sl                                # key
            child = self._u(p, self.so); p += self.so
            if level > 0:
                self._walk_btree(child, heap, kids)
            else:
                if b[child:child + 4] != b'SNOD':
                    raise H5Error('bad symbol node')
                count = self._u(child + 6, 2)
                q = child + 8
                for _ in range(count):
                    e = self._sym_entry(q)
                    kids[self._heap_string(heap, e['name_off'])] = e
                    q += 2 * self.so + 24

    # ── datasets ───────────────────────────────────────────────────────
    def _dataset(self, header_addr):
        shape = dtype = raw = None
        for mtype, body, size in self._messages(header_addr):
            b = self.b
            if mtype == 0x01:                           # dataspace
                ver, rank = b[body], b[body + 1]
                p = body + (8 if ver == 1 else 4)
                shape = tuple(self._u(p + i * self.sl, self.sl) for i in range(rank))
            elif mtype == 0x03:                         # datatype
                cls, nbytes = b[body] & 0x0F, self._u(body + 4, 4)
                big = b[body + 1] & 1
                if cls == 1:
                    dtype = np.dtype(f'{">" if big else "<"}f{nbytes}')
                elif cls == 0:
                    signed = (b[body + 1] >> 3) & 1
                    dtype = np.dtype(f'{">" if big else "<"}{"i" if signed else "u"}{nbytes}')
                else:
                    raise H5Error(f'datatype class {cls} not supported')
            elif mtype == 0x08:                         # data layout
                ver, cls = b[body], b[body + 1]
                if ver != 3:
                    raise H5Error(f'layout v{ver} not supported')
                if cls == 0:
                    n = self._u(body + 2, 2); raw = b[body + 4: body + 4 + n]
                elif cls == 1:
                    a, n = self._u(body + 2, self.so), self._u(body + 2 + self.so, self.sl)
                    raw = b[a:a + n] if a != (1 << (8 * self.so)) - 1 else b''
                else:
                    raise H5Error('chunked/compressed datasets not supported')
        if shape is None or dtype is None or raw is None:
            return None                                 # not a dataset (a group)
        count = int(np.prod(shape)) if shape else 1
        arr = np.frombuffer(raw, dtype=dtype, count=count) if raw else np.zeros(count, dtype)
        return arr.reshape(shape).astype(dtype.newbyteorder('<'))

    def walk(self):
        """Yield (path, ndarray) for every dataset in the file."""
        def rec(entry, path):
            arr = self._dataset(entry['header']) if entry['btree'] is None else None
            if arr is not None:
                yield path, arr
                return
            for name, e in sorted(self._children(entry['header'], entry['btree'], entry['heap']).items()):
                yield from rec(e, f'{path}/{name}' if path else name)
        yield from rec(self.root, '')
