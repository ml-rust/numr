#!/usr/bin/env python3
"""Write the hard-to-round fixtures of tests/correctly_rounded from CORE-MATH.

Usage: port_hard_cases.py <core-math checkout> <output directory>

Reads src/binary64/{exp/exp.wc,log/log.wc,log1p/log1p.wc} and writes
{exp,log,log1p}.bin. Each file holds the sorted, distinct binary64 bit
patterns of the finite inputs, as unsigned LEB128 deltas from the previous
pattern (the first from 0). Non-finite entries (nan, snan, inf) are left out:
the special-value tests cover them.
"""

import math
import struct
import sys

FUNCTIONS = ["exp", "log", "log1p"]


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
            value = float.fromhex(token)
            if not math.isfinite(value):
                raise SystemExit(f"{path}: unexpected token {token}")
            if float.fromhex(value.hex()) != value:
                raise SystemExit(f"{path}: inexact token {token}")
            bits.add(struct.unpack("<Q", struct.pack("<d", value))[0])
    return sorted(bits)


def encode(values):
    out = bytearray()
    previous = 0
    for value in values:
        delta = value - previous
        previous = value
        while True:
            byte = delta & 0x7F
            delta >>= 7
            if delta:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
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


if __name__ == "__main__":
    main()
