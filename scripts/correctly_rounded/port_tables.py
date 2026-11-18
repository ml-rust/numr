#!/usr/bin/env python3
"""Write src/algorithm/correctly_rounded/tables_data.rs from CORE-MATH C sources.

Usage: port_tables.py <core-math checkout> <commit> <output .rs>

The checkout must hold src/binary64/{exp/exp.c,log/log.c,log/dint.h,log1p/log1p.c}
at <commit>. Every double literal is parsed exactly and rejected if binary64
cannot hold it. Each table becomes one flat `pub static NAME: [u64; LEN]`.
"""

import re
import struct
import sys
from fractions import Fraction

MIT_NOTICE = """\
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE."""

# (source file, C array name, occurrence index, Rust name, expected length, doc)
DOUBLE_TABLES = [
    ("exp/exp.c", "db", 0, "EXP_EXCEPTIONS", 63,
     "`exp` inputs whose accurate path misrounds: `(x, h, l)` per entry, sorted by `x`. "
     "The result is `(h + l) * 2^ie`."),
    ("exp/exp.c", "t0", 0, "EXP_T0", 128,
     "`(lo, hi)` per entry: `2^(i/2^6)` for `0 <= i < 64`."),
    ("exp/exp.c", "t1", 0, "EXP_T1", 128,
     "`(lo, hi)` per entry: `2^(i/2^12)` for `0 <= i < 64`."),
    ("exp/exp.c", "ch", 0, "EXP_ACCURATE_POLY", 14,
     "`(hi, lo)` per entry: degree 1 to 7 coefficients of `e^x - 1` for the accurate path."),
    ("exp/exp.c", "ch", 1, "EXP_FAST_POLY", 4,
     "Fast-path coefficients `c0..c3` of `(e^x - 1) / x`."),
    ("log/log.c", "_INVERSE", 0, "LOG_INVERSE", 363,
     "`r[i - 362]`: a 10-bit approximation of `1/y` for `i * 2^-9 <= y < (i + 1) * 2^-9`, "
     "`362 <= i <= 724`."),
    ("log/log.c", "_LOG_INV", 0, "LOG_LOG_INV", 726,
     "`(h, l)` per entry: `-log(LOG_INVERSE[i])`, `h` a multiple of `2^-42`."),
    ("log/log.c", "P", 0, "LOG_FAST_POLY", 6,
     "Degree 1 to 6 coefficients of `log(1 + z)` for the fast path."),
    ("log/log.c", "c", 0, "LOG_TINY_POLY", 6,
     "Degree 2 to 7 coefficients of `log(1 + r)` for `|r| <= 2^-9`."),
    ("log1p/log1p.c", "rt", 0, "LOG1P_RT", 64,
     "`rt[k][j]` at `16 * k + j`: the four reciprocal tables of the accurate path."),
    ("log1p/log1p.c", "ln", 0, "LOG1P_LN", 192,
     "`(l0, l1, l2)` at `3 * (16 * k + j)`: `-log(rt[k][j]) = l2 + l1 + l0`."),
    ("log1p/log1p.c", "rf", 0, "LOG1P_RF", 64,
     "Fast-path reciprocals, about `1/sqrt(2)` to `sqrt(2)`."),
    ("log1p/log1p.c", "lf", 0, "LOG1P_LF", 128,
     "`(l0, l1)` per entry: `-log(LOG1P_RF[j]) = l1 + l0`."),
    ("log1p/log1p.c", "c", 0, "LOG1P_TINY_POLY", 2,
     "Coefficients for `|x| < 0x1.19bp-21`."),
    ("log1p/log1p.c", "c", 1, "LOG1P_SMALL_POLY", 4,
     "Coefficients for `0x1.19bp-21 <= |x| < 2^-12`."),
    ("log1p/log1p.c", "c", 2, "LOG1P_MEDIUM_POLY", 12,
     "Coefficients for `2^-12 <= |x| < 2^-4`."),
    ("log1p/log1p.c", "c", 3, "LOG1P_LARGE_POLY", 6,
     "Coefficients for `|x| >= 2^-4`."),
    ("log1p/log1p.c", "cz", 0, "LOG1P_REFINE_CZ", 10,
     "`(hi, lo)` per entry: accurate-path coefficients near zero."),
    ("log1p/log1p.c", "czl", 0, "LOG1P_REFINE_CZL", 4,
     "Accurate-path low-order coefficients near zero."),
    ("log1p/log1p.c", "cy", 0, "LOG1P_REFINE_CY", 10,
     "`(hi, lo)` per entry: accurate-path coefficients after reduction."),
    ("log1p/log1p.c", "cl", 0, "LOG1P_REFINE_CL", 4,
     "Accurate-path low-order coefficients after reduction."),
]

# (C array name, Rust name, expected entry count or None, doc)
DINT_TABLES = [
    ("_INVERSE_2", "LOG_DINT_INVERSE", None,
     "Accurate-path reciprocals: `(hi, lo, ex, sgn)` per entry, value "
     "`(-1)^sgn * (hi + lo / 2^64) * 2^(ex - 63)`."),
    ("_LOG_INV_2", "LOG_DINT_LOG_INV", None,
     "`-log` of each `LOG_DINT_INVERSE` entry, same layout."),
    ("P_2", "LOG_DINT_POLY", 13,
     "Accurate-path polynomial for `log(1 + z)`, highest degree first, same layout."),
]

# (C scalar name, Rust name, doc)
DINT_SCALARS = [
    ("M_ONE", "LOG_DINT_MINUS_ONE", "`-1` in the dint layout."),
    ("LOG2", "LOG_DINT_LN2", "`log(2)` in the dint layout, absolute error below `2^-129.97`."),
]

HEX_FLOAT = re.compile(r"[-+]?0x[0-9a-fA-F]*\.?[0-9a-fA-F]*p[-+]?\d+")
DEC_FLOAT = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")
TOKEN = re.compile(HEX_FLOAT.pattern + "|" + DEC_FLOAT.pattern)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def exact_value(token):
    sign = -1 if token.startswith("-") else 1
    body = token.lstrip("+-")
    if body.lower().startswith("0x"):
        mant, exp = body[2:].lower().split("p")
        whole, _, frac = mant.partition(".")
        digits = (whole or "0") + frac
        value = Fraction(int(digits, 16), 16 ** len(frac))
        value *= Fraction(2) ** int(exp)
    else:
        value = Fraction(body)
    return sign * value


def to_bits(token):
    exact = exact_value(token)
    as_float = float(exact)
    if Fraction(as_float) != exact:
        raise SystemExit(f"literal {token} is not a binary64 value")
    if as_float == 0 and token.lstrip("+").startswith("-"):
        as_float = -0.0
    return struct.unpack("<Q", struct.pack("<d", as_float))[0]


def array_body(text, name, occurrence):
    pattern = re.compile(r"static const double\s+" + re.escape(name) + r"\s*(\[[^\]]*\])*\s*=")
    matches = list(pattern.finditer(text))
    if len(matches) <= occurrence:
        raise SystemExit(f"array {name} occurrence {occurrence} not found")
    start = text.index("{", matches[occurrence].end())
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos + 1]
    raise SystemExit(f"array {name} is not closed")


def dint_entries(body):
    entry = re.compile(
        r"\.hi\s*=\s*(0x[0-9a-fA-F]+)\s*,\s*\.lo\s*=\s*(0x[0-9a-fA-F]+)\s*,"
        r"\s*\.ex\s*=\s*(-?\d+)\s*,\s*\.sgn\s*=\s*(0x[0-9a-fA-F]+)")
    words = []
    for hi, lo, ex, sgn in entry.findall(body):
        words += [int(hi, 16), int(lo, 16), int(ex) & 0xFFFF_FFFF_FFFF_FFFF, int(sgn, 16)]
    return words


def dint_body(text, name, is_array):
    marker = f"static const dint64_t {name}" + ("[] = {" if is_array else " = {")
    start = text.index(marker) + len(marker) - 1
    end = text.index("};", start)
    return text[start:end + 1]


def emit(out, name, words, doc, per_line):
    out.append(f"/// {doc}")
    out.append("#[rustfmt::skip]")
    out.append(f"pub static {name}: [u64; {len(words)}] = [")
    for pos in range(0, len(words), per_line):
        row = ", ".join(f"0x{w:016x}" for w in words[pos:pos + per_line])
        out.append(f"    {row},")
    out.append("];")
    out.append("")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    root, commit, target = sys.argv[1], sys.argv[2], sys.argv[3]
    base = f"{root}/src/binary64"
    sources = {}
    out = [
        f"// @generated by scripts/correctly_rounded/port_tables.py from CORE-MATH commit {commit}",
        "// (https://gitlab.inria.fr/core-math/core-math), files src/binary64/exp/exp.c,",
        "// src/binary64/log/log.c, src/binary64/log/dint.h and src/binary64/log1p/log1p.c.",
        "// Do not edit by hand: rerun the generator.",
        "//",
        "// exp.c: Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.",
        "// log1p.c: Copyright (c) 2024-2026 Alexei Sibidanov <sibid@uvic.ca>.",
        "// log.c: Copyright (c) 2022-2026 INRIA and CERN. Authors: Paul Zimmermann and Tom Hubrecht.",
        "// dint.h: Copyright (c) 2022 CERN. Author: Tom Hubrecht.",
        "// This file is part of the CORE-MATH project (https://core-math.gitlabpages.inria.fr/).",
        "//",
        MIT_NOTICE,
        "",
    ]
    for path, cname, occurrence, rname, length, doc in DOUBLE_TABLES:
        if path not in sources:
            with open(f"{base}/{path}") as handle:
                sources[path] = strip_comments(handle.read())
        body = array_body(sources[path], cname, occurrence)
        words = [to_bits(tok) for tok in TOKEN.findall(body)]
        if len(words) != length:
            raise SystemExit(f"{rname}: {len(words)} values, expected {length}")
        emit(out, rname, words, doc, 3 if rname == "EXP_EXCEPTIONS" or rname == "LOG1P_LN" else 4)
    with open(f"{base}/log/dint.h") as handle:
        dint = strip_comments(handle.read())
    for cname, rname, doc in DINT_SCALARS:
        words = dint_entries(dint_body(dint, cname, False))
        if len(words) != 4:
            raise SystemExit(f"{rname}: malformed scalar")
        emit(out, rname, words, doc, 4)
    for cname, rname, count, doc in DINT_TABLES:
        words = dint_entries(dint_body(dint, cname, True))
        if count is not None and len(words) != 4 * count:
            raise SystemExit(f"{rname}: {len(words) // 4} entries, expected {count}")
        emit(out, rname, words, doc, 4)
    with open(target, "w") as handle:
        handle.write("\n".join(out).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
