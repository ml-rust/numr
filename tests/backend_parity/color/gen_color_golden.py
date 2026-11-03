#!/usr/bin/env python3
"""Write color_golden.bin: numpy references for ColorOps.

Run from any directory: python3 tests/backend_parity/color/gen_color_golden.py

The references come from a numpy implementation of the formulas documented
on numr's ColorOps trait, NOT from swscale. Every step is one float32
operation in the documented order, so the references match numr's kernels
bit for bit. Coefficients are computed in float64 with the same expressions
as ColorCoeffs::new and rounded to float32 once.

When ffmpeg is on PATH, one extra block holds a BT.709 limited NV12 frame and
ffmpeg's rgb24 conversion of it. The Rust test bounds the filter difference
with PSNR > 40 dB. Without ffmpeg that block is absent, and the script says
so on stderr.

File layout, little endian:

    b"NCLR", u32 version (1)
    u32 decode count, then per block:
        u32 format, matrix, range, siting, h, w
        planes (see below), then 3 * h * w f32: RGB in [3, H, W] order
    u32 encode count, then per block:
        u32 format, matrix, range, siting, h, w
        3 * h * w f32 RGB input in [3, H, W] order, then planes
    u32 tie count, then per tie: f32 x, u32 want
    u32 ffmpeg flag (0 or 1), and when 1:
        u32 h, w, then y (h * w u8), uv (h / 2 * w u8), rgb24 (h * w * 3 u8)

planes: y [h, w], then for NV12 and P010 uv [h/2, w] interleaved U, V, and for
I420 u [h/2, w/2] then v [h/2, w/2]. NV12 and I420 samples are u8, P010
samples are u16 with the value in the high 10 bits.

format: 0 NV12, 1 I420, 2 P010. matrix: 0 BT.601, 1 BT.709.
range: 0 limited, 1 full. siting: 0 left, 1 center.
"""

import math
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_golden.bin")
F = np.float32
H, W = 6, 8
FF_H, FF_W = 48, 64


def coeffs(matrix, rng, bits):
    kr, kb = (0.299, 0.114) if matrix == 0 else (0.2126, 0.0722)
    kg = 1.0 - kr - kb
    s = float(1 << (bits - 8))
    mx = float((1 << bits) - 1)
    if rng == 0:
        y_off, c_off, y_mul, c_mul = 16.0 * s, 128.0 * s, 219.0 * s, 224.0 * s
        y_lo, y_hi, c_lo, c_hi = 16.0 * s, 235.0 * s, 16.0 * s, 240.0 * s
    else:
        y_off, c_off, y_mul, c_mul = 0.0, (mx + 1.0) / 2.0, mx, mx
        y_lo, y_hi, c_lo, c_hi = 0.0, mx, 0.0, mx
    return {
        "y_off": int(y_off),
        "c_off": int(c_off),
        "c_div16": F(16.0 * c_mul),
        "cr_r": F(2.0 * (1.0 - kr)),
        "cb_b": F(2.0 * (1.0 - kb)),
        "g_cb": F(-2.0 * kb * (1.0 - kb) / kg),
        "g_cr": F(-2.0 * kr * (1.0 - kr) / kg),
        "kr": F(kr),
        "kg": F(kg),
        "kb": F(kb),
        "y_mul": F(y_mul),
        "c_mul": F(c_mul),
        "y_lo": F(y_lo),
        "y_hi": F(y_hi),
        "c_lo": F(c_lo),
        "c_hi": F(c_hi),
    }


def clamp01(x):
    return np.fmin(np.fmax(F(x), F(0.0)), F(1.0))


def up_taps(pos, n, center):
    k = pos // 2
    nxt = min(k + 1, n - 1)
    even = pos % 2 == 0
    if center:
        return ([max(k - 1, 0), k], [1, 3]) if even else ([k, nxt], [3, 1])
    return ([k, k], [4, 0]) if even else ([k, nxt], [2, 2])


def decode(k, y, u, v, center):
    """y [H, W], u, v [H/2, W/2] integer values -> RGB float32 [3, H, W]."""
    h, w = y.shape
    out = np.zeros((3, h, w), dtype=np.float32)
    for row in range(h):
        rows, wv = up_taps(row, h // 2, True)
        for col in range(w):
            cols, wh = up_taps(col, w // 2, center)
            cb16 = cr16 = 0
            for a in range(2):
                for b in range(2):
                    wt = wv[a] * wh[b]
                    cb16 += wt * int(u[rows[a], cols[b]])
                    cr16 += wt * int(v[rows[a], cols[b]])
            yn = F(int(y[row, col]) - k["y_off"]) / k["y_mul"]
            cb = F(cb16 - 16 * k["c_off"]) / k["c_div16"]
            cr = F(cr16 - 16 * k["c_off"]) / k["c_div16"]
            r = yn + k["cr_r"] * cr
            bb = yn + k["cb_b"] * cb
            g = (yn + k["g_cb"] * cb) + k["g_cr"] * cr
            out[:, row, col] = [clamp01(r), clamp01(g), clamp01(bb)]
    return out


def quantize(v, mul, off, lo, hi):
    return np.rint(np.fmin(np.fmax(F(v) * mul + F(off), lo), hi))


def encode_pixel(k, rgb, row, col):
    r, g, b = (clamp01(rgb[c, row, col]) for c in range(3))
    y = (k["kr"] * r + k["kg"] * g) + k["kb"] * b
    return y, (b - y) / k["cb_b"], (r - y) / k["cr_r"]


def encode(k, rgb, center):
    """RGB float32 [3, H, W] -> integer codes y [H, W], u, v [H/2, W/2]."""
    _, h, w = rgb.shape
    y = np.zeros((h, w), dtype=np.int64)
    u = np.zeros((h // 2, w // 2), dtype=np.int64)
    v = np.zeros((h // 2, w // 2), dtype=np.int64)
    for i in range(h // 2):
        for j in range(w // 2):
            cbs, crs = [], []
            for row in (2 * i, 2 * i + 1):
                cb = [F(0), F(0), F(0)]
                cr = [F(0), F(0), F(0)]
                for slot, col in ((1, 2 * j), (2, 2 * j + 1)):
                    yv, cb[slot], cr[slot] = encode_pixel(k, rgb, row, col)
                    y[row, col] = int(quantize(yv, k["y_mul"], k["y_off"], k["y_lo"], k["y_hi"]))
                if not center:
                    _, cb[0], cr[0] = encode_pixel(k, rgb, row, max(2 * j - 1, 0))
                cbs.append(cb)
                crs.append(cr)

            def down(t, b):
                if center:
                    return ((t[1] + t[2]) + (b[1] + b[2])) * F(0.25)
                ht = (t[0] + F(2.0) * t[1]) + t[2]
                hb = (b[0] + F(2.0) * b[1]) + b[2]
                return (ht + hb) * F(0.125)

            args = (k["c_mul"], k["c_off"], k["c_lo"], k["c_hi"])
            u[i, j] = int(quantize(down(cbs[0], cbs[1]), *args))
            v[i, j] = int(quantize(down(crs[0], crs[1]), *args))
    return y, u, v


def plane_bytes(fmt, y, u, v, low_bits=None):
    """Stored plane bytes of integer codes. P010 shifts left by 6 and ORs in
    `low_bits` when given, so readers must drop them."""
    if fmt == 2:
        shift = lambda a: (a.astype(np.uint16) << 6)
        ys, us, vs = shift(y), shift(u), shift(v)
        if low_bits is not None:
            ys = ys | low_bits[0]
            us = us | low_bits[1]
            vs = vs | low_bits[2]
        uv = np.stack([us, vs], axis=-1).reshape(us.shape[0], -1)
        return ys.astype("<u2").tobytes() + uv.astype("<u2").tobytes()
    ys, us, vs = (a.astype(np.uint8) for a in (y, u, v))
    if fmt == 0:
        uv = np.stack([us, vs], axis=-1).reshape(us.shape[0], -1)
        return ys.tobytes() + uv.tobytes()
    return ys.tobytes() + us.tobytes() + vs.tobytes()


def specs():
    for fmt in (0, 1, 2):
        for matrix in (0, 1):
            for rng in (0, 1):
                for siting in (0, 1):
                    yield fmt, matrix, rng, siting


def decode_blocks(gen):
    blocks = []
    for fmt, matrix, rng, siting in specs():
        bits = 10 if fmt == 2 else 8
        top = (1 << bits) - 1
        y = gen.integers(0, top + 1, size=(H, W))
        u = gen.integers(0, top + 1, size=(H // 2, W // 2))
        v = gen.integers(0, top + 1, size=(H // 2, W // 2))
        low = None
        if fmt == 2:
            low = tuple(
                gen.integers(0, 64, size=a.shape).astype(np.uint16) for a in (y, u, v)
            )
        rgb = decode(coeffs(matrix, rng, bits), y, u, v, siting == 1)
        header = struct.pack("<6I", fmt, matrix, rng, siting, H, W)
        payload = plane_bytes(fmt, y, u, v, low) + rgb.astype("<f4").tobytes()
        blocks.append(header + payload)
    return blocks


def encode_blocks(gen):
    blocks = []
    for fmt, matrix, rng, siting in specs():
        bits = 10 if fmt == 2 else 8
        rgb = gen.uniform(-0.1, 1.1, size=(3, H, W)).astype(np.float32)
        y, u, v = encode(coeffs(matrix, rng, bits), rgb, siting == 1)
        header = struct.pack("<6I", fmt, matrix, rng, siting, H, W)
        payload = rgb.astype("<f4").tobytes() + plane_bytes(fmt, y, u, v)
        blocks.append(header + payload)
    return blocks


def ties():
    """f32 inputs x with x * 255 exactly k + 0.5 in f32, and the even code."""
    out = []
    for k in range(255):
        x = F((2 * k + 1) / 510.0)
        for _ in range(4):
            x = np.nextafter(x, F(0.0))
        for _ in range(9):
            if x * F(255.0) == F(k + 0.5):
                want = int(np.rint(clamp01(x) * F(255.0)))
                assert want == k + (k % 2), (k, x, want)
                out.append((x, want))
                break
            x = np.nextafter(x, F(1.0))
    return out


def ffmpeg_block():
    exe = shutil.which("ffmpeg")
    if exe is None:
        print("ffmpeg not on PATH: no ffmpeg block written", file=sys.stderr)
        return None, None
    rr, cc = np.meshgrid(np.arange(FF_H), np.arange(FF_W), indexing="ij")
    rgb = np.stack(
        [
            0.5 + 0.4 * np.sin(2 * math.pi * cc / FF_W + 0.3),
            0.5 + 0.4 * np.cos(2 * math.pi * rr / FF_H),
            0.5 + 0.3 * np.sin(2 * math.pi * (cc + rr) / (FF_W + FF_H)),
        ]
    ).astype(np.float32)
    k = coeffs(1, 0, 8)
    y, u, v = encode(k, rgb, False)
    nv12 = plane_bytes(0, y, u, v)
    vf = (
        "scale=in_color_matrix=bt709:in_range=tv:in_chroma_loc=left:"
        "flags=bilinear+accurate_rnd+full_chroma_int,format=rgb24"
    )
    cmd = [
        exe, "-v", "error", "-f", "rawvideo", "-pix_fmt", "nv12",
        "-s", f"{FF_W}x{FF_H}", "-i", "pipe:0", "-vf", vf,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    ref = subprocess.run(cmd, input=nv12, capture_output=True, check=True).stdout
    assert len(ref) == FF_H * FF_W * 3, len(ref)
    ours = decode(k, y, u, v, False)
    ours24 = np.rint(np.fmin(np.fmax(ours, F(0)), F(1)) * F(255)).astype(np.uint8)
    ours24 = ours24.transpose(1, 2, 0).reshape(-1)
    theirs = np.frombuffer(ref, dtype=np.uint8)
    mse = np.mean((ours24.astype(np.float64) - theirs.astype(np.float64)) ** 2)
    psnr = float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)
    header = struct.pack("<3I", 1, FF_H, FF_W)
    return header + nv12 + ref, psnr


def main():
    gen = np.random.default_rng(20261104)
    dec = decode_blocks(gen)
    enc = encode_blocks(gen)
    tie = ties()
    ff, psnr = ffmpeg_block()
    with open(OUT, "wb") as f:
        f.write(b"NCLR" + struct.pack("<I", 1))
        for blocks in (dec, enc):
            f.write(struct.pack("<I", len(blocks)))
            for b in blocks:
                f.write(b)
        f.write(struct.pack("<I", len(tie)))
        for x, want in tie:
            f.write(struct.pack("<fI", x, want))
        f.write(ff if ff is not None else struct.pack("<I", 0))
    note = "no ffmpeg block" if ff is None else f"ffmpeg PSNR {psnr:.2f} dB"
    print(
        f"wrote {len(dec)} decode, {len(enc)} encode, {len(tie)} tie blocks, {note}, "
        f"{os.path.getsize(OUT)} bytes to {OUT} (numpy {np.__version__})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
