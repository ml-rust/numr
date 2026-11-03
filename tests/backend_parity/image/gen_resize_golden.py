#!/usr/bin/env python3
"""Write resize_golden.bin: PyTorch CPU references for ImageOps::resize.

Run from any directory: python3 tests/backend_parity/image/gen_resize_golden.py

Every reference is torch.nn.functional.interpolate on CPU in float32 with
size=(oh, ow) and align_corners=False. The input is not stored. Both sides
build it from its NCHW element index i:

    k = (i * 7919 + 104729) % 4099
    x = f32(k) / f32(4099) - f32(0.25)

which gives values in [-0.25, 0.75). Kind 1 and kind 2 blocks round that
input to f16 or bf16 (round to nearest even) and take the float32 reference
from the rounded values.

File layout, little endian:

    b"NRSZ", u32 version (1), u32 block count
    per block: u32 kind, n, c, h, w, oh, ow, filter, antialias
               then n * c * oh * ow f32 values in NCHW order

kind: 0 f32 input, 1 f16-rounded input, 2 bf16-rounded input.
filter: 0 bilinear, 1 bicubic.
"""

import os
import struct
import sys

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resize_golden.bin")

ALL_MODES = [(f, aa) for f in (0, 1) for aa in (0, 1)]

# (kind, input NCHW, output (oh, ow), modes)
CASES = [
    (0, (1, 1, 16, 24), (8, 12), ALL_MODES),  # exact 2x downsample
    (0, (1, 1, 3, 4), (6, 8), ALL_MODES),  # exact 2x upsample
    (0, (1, 1, 17, 23), (30, 11), ALL_MODES),  # non-integer, H up, W down
    (0, (1, 1, 30, 11), (17, 23), ALL_MODES),  # non-integer, H down, W up
    (0, (1, 1, 64, 3), (63, 2), ALL_MODES),  # ratio near 1 along H
    (0, (1, 1, 64, 64), (5, 5), ALL_MODES),  # extreme downsample
    (0, (1, 1, 7, 9), (1, 1), ALL_MODES),  # size 1 output
    (0, (1, 1, 1, 1), (5, 4), ALL_MODES),  # size 1 input
    (0, (2, 3, 8, 9), (5, 7), ALL_MODES),  # batch 2, 3 channels
    (0, (1, 1, 64, 64), (32, 32), [(1, 1)]),  # SRVGG 4x -> 2x: exact /2, bicubic AA
    (1, (2, 3, 8, 9), (5, 7), ALL_MODES),  # f16-rounded input
    (2, (2, 3, 8, 9), (5, 7), ALL_MODES),  # bf16-rounded input
]


def make_input(shape):
    i = np.arange(int(np.prod(shape)), dtype=np.int64)
    k = ((i * 7919 + 104729) % 4099).astype(np.float32)
    return (k / np.float32(4099) - np.float32(0.25)).astype(np.float32).reshape(shape)


def rounded(x, kind):
    t = torch.from_numpy(x)
    if kind == 1:
        return t.to(torch.float16).to(torch.float32)
    if kind == 2:
        return t.to(torch.bfloat16).to(torch.float32)
    return t


def main():
    blocks = []
    for kind, shape, (oh, ow), modes in CASES:
        x = rounded(make_input(shape), kind)
        for filt, aa in modes:
            y = torch.nn.functional.interpolate(
                x,
                size=(oh, ow),
                mode="bicubic" if filt else "bilinear",
                align_corners=False,
                antialias=bool(aa),
            )
            header = struct.pack("<9I", kind, *shape, oh, ow, filt, aa)
            payload = y.contiguous().numpy().astype("<f4").tobytes()
            blocks.append(header + payload)
    with open(OUT, "wb") as f:
        f.write(b"NRSZ" + struct.pack("<II", 1, len(blocks)))
        for b in blocks:
            f.write(b)
    print(
        f"wrote {len(blocks)} blocks, {os.path.getsize(OUT)} bytes to {OUT} "
        f"(torch {torch.__version__}, CPU capability "
        f"{torch.backends.cpu.get_cpu_capability()})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
