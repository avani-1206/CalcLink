"""
Minimal ONNX file writer (no `onnx` package needed).

An .onnx file is just a Protocol Buffers message (ModelProto).  Protobuf's wire
format is simple: every field is  <tag = field_number<<3 | wire_type> <value>,
where wire_type 0 = varint, 2 = length-delimited bytes, 5 = 32-bit float.
We encode only the handful of fields our small CNN needs.
"""
import struct
import numpy as np


def _varint(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field, wt):
    return _varint((field << 3) | wt)


def f_int(field, v):
    return _key(field, 0) + _varint(int(v))


def f_bytes(field, b):
    if isinstance(b, str):
        b = b.encode()
    return _key(field, 2) + _varint(len(b)) + b


def f_float(field, v):
    return _key(field, 5) + struct.pack('<f', v)


# ── ONNX messages ───────────────────────────────────────────────────────────
def tensor(name, arr):                         # TensorProto
    arr = np.ascontiguousarray(arr, np.float32)
    b = b''.join(f_int(1, d) for d in arr.shape)  # dims
    b += f_int(2, 1)                               # data_type = FLOAT
    b += f_bytes(8, name)
    b += f_bytes(9, arr.tobytes())                 # raw_data (little endian)
    return b


def attr_ints(name, vals):                    # AttributeProto, type INTS = 7
    return f_bytes(1, name) + b''.join(f_int(8, v) for v in vals) + f_int(20, 7)


def attr_int(name, v):                        # type INT = 2
    return f_bytes(1, name) + f_int(3, v) + f_int(20, 2)


def node(op, inputs, outputs, attrs=()):      # NodeProto
    b = b''.join(f_bytes(1, i) for i in inputs)
    b += b''.join(f_bytes(2, o) for o in outputs)
    b += f_bytes(3, outputs[0] + '_' + op)
    b += f_bytes(4, op)
    b += b''.join(f_bytes(5, a) for a in attrs)
    return b


def value_info(name, dims):                   # ValueInfoProto (float tensor)
    shape = b''
    for d in dims:
        dim = f_bytes(2, d) if isinstance(d, str) else f_int(1, d)
        shape += f_bytes(1, dim)
    tensor_type = f_int(1, 1) + f_bytes(2, shape)
    return f_bytes(1, name) + f_bytes(2, f_bytes(1, tensor_type))


def save_onnx(P, path, n_classes):
    nodes = [
        node('Conv', ['input', 'W1', 'b1'], ['c1'], [attr_ints('kernel_shape', [3, 3]), attr_ints('pads', [1, 1, 1, 1])]),
        node('Relu', ['c1'], ['r1']),
        node('MaxPool', ['r1'], ['p1'], [attr_ints('kernel_shape', [2, 2]), attr_ints('strides', [2, 2])]),
        node('Conv', ['p1', 'W2', 'b2'], ['c2'], [attr_ints('kernel_shape', [3, 3]), attr_ints('pads', [1, 1, 1, 1])]),
        node('Relu', ['c2'], ['r2']),
        node('MaxPool', ['r2'], ['p2'], [attr_ints('kernel_shape', [2, 2]), attr_ints('strides', [2, 2])]),
        node('Flatten', ['p2'], ['f'], [attr_int('axis', 1)]),
        node('Gemm', ['f', 'W3', 'b3'], ['h1']),
        node('Relu', ['h1'], ['h']),
        node('Gemm', ['h', 'W4', 'b4'], ['logits']),
        node('Softmax', ['logits'], ['probs'], [attr_int('axis', 1)]),
    ]
    graph = b''.join(f_bytes(1, n) for n in nodes)
    graph += f_bytes(2, 'calcink_symbols')
    graph += b''.join(f_bytes(5, tensor(k, P[k])) for k in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'])
    graph += f_bytes(11, value_info('input', ['N', 1, 28, 28]))
    graph += f_bytes(12, value_info('probs', ['N', n_classes]))
    model = f_int(1, 7)                              # ir_version 7
    model += f_bytes(2, 'calcink-numpy')              # producer_name
    model += f_bytes(7, graph)
    model += f_bytes(8, f_bytes(1, '') + f_int(2, 13))  # opset_import: default domain, v13
    with open(path, 'wb') as fh:
        fh.write(model)
