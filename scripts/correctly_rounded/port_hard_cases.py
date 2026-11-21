#!/usr/bin/env python3
"""Write the hard-to-round fixtures of tests/correctly_rounded from CORE-MATH.

Usage: port_hard_cases.py <core-math checkout> <output directory>

Reads src/binary64/{acos/acos.wc,exp/exp.wc,log/log.wc,log1p/log1p.wc} and
writes {acos,exp,log,log1p}.bin. Each file holds the sorted, distinct binary64 bit
patterns of the finite inputs, as unsigned LEB128 deltas from the previous
pattern (the first from 0). Non-finite entries (nan, snan, inf) are left out:
the special-value tests cover them.

Reads src/binary64/pow/pow.wc, whose entries are "x,y" pairs, and writes
pow.bin. The distinct finite pairs are grouped by the bit pattern of y, in
ascending order. Each group is the LEB128 delta of y from the previous group's
y (the first from 0), the LEB128 count of its pairs, then the sorted bit
patterns of x as LEB128 deltas (the first from 0). Pairs with a non-finite
operand are left out.
"""

import math
import struct
import sys

FUNCTIONS = ["acos", "exp", "log", "log1p"]


def inputs(path):
    bits = set()
    with open(path) as handle:
        for line in handle:
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            token = text.split()[0]
            if "nan" in token or "inf" in token:
                continue
            bits.add(finite_bits(token, path))
    return sorted(bits)


def finite_bits(token, path):
    value = float.fromhex(token)
    if not math.isfinite(value):
        raise SystemExit(f"{path}: unexpected token {token}")
    if float.fromhex(value.hex()) != value:
        raise SystemExit(f"{path}: inexact token {token}")
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def pairs(path):
    groups = {}
    with open(path) as handle:
        for line in handle:
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            x_token, y_token = (t.strip() for t in text.split(",")[:2])
            if any("nan" in t or "inf" in t for t in (x_token, y_token)):
                continue
            x, y = finite_bits(x_token, path), finite_bits(y_token, path)
            groups.setdefault(y, set()).add(x)
    return groups


def leb128(out, value):
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def encode(values):
    out = bytearray()
    previous = 0
    for value in values:
        leb128(out, value - previous)
        previous = value
    return bytes(out)


def encode_pairs(groups):
    out = bytearray()
    previous_y = 0
    for y in sorted(groups):
        leb128(out, y - previous_y)
        previous_y = y
        xs = sorted(groups[y])
        leb128(out, len(xs))
        out += encode(xs)
    return bytes(out)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    root, target = sys.argv[1], sys.argv[2]
    for name in FUNCTIONS:
        values = inputs(f"{root}/src/binary64/{name}/{name}.wc")
        with open(f"{target}/{name}.bin", "wb") as handle:
            handle.write(encode(values))
        print(f"{name}: {len(values)} inputs")
    groups = pairs(f"{root}/src/binary64/pow/pow.wc")
    with open(f"{target}/pow.bin", "wb") as handle:
        handle.write(encode_pairs(groups))
    print(f"pow: {sum(len(xs) for xs in groups.values())} pairs")


if __name__ == "__main__":
    main()
