#!/usr/bin/env python3
"""Write src/algorithm/correctly_rounded/tables_data.rs from CORE-MATH C sources.

Usage: port_tables.py <core-math checkout> <commit> <output .rs>

The checkout must hold src/binary64/{exp/exp.c,log/log.c,log/dint.h,log1p/log1p.c,
pow/pow.h,pow/dint.h,pow/qint.h,acos/acos.c} at <commit>. Every double literal is parsed
exactly and rejected if binary64 cannot hold it. Each table becomes one flat
`pub static NAME: [u64; LEN]`.

A pow table that equals one already emitted is checked and not emitted again:
pow's `LOG2` and `M_ONE` equal log's, and the 256-bit reciprocal tables
`_INVERSE_3_*`, `LOG2_INV_Q` and `M_ONE_Q` equal their 128-bit forms.
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
    ("pow/pow.h", "_INVERSE", 0, "POW_INVERSE", 182,
     "`r[i - 181]`: a 9-bit approximation of `1/y` for `i * 2^-8 <= y < (i + 1) * 2^-8`, "
     "`181 <= i <= 362`. `r * y - 1` is exact."),
    ("pow/pow.h", "_LOG_INV", 0, "POW_LOG_INV", 364,
     "`(h, l)` per entry: `-log(POW_INVERSE[i])`, `h` a multiple of `2^-42`."),
    ("pow/pow.h", "T1", 0, "POW_T1", 128,
     "`(hi, lo)` per entry: `2^(i/2^6)` for `0 <= i < 64`, error below `2^-107`."),
    ("pow/pow.h", "T2", 0, "POW_T2", 128,
     "`(hi, lo)` per entry: `2^(i/2^12)` for `0 <= i < 64`, error below `2^-107`."),
    ("pow/pow.h", "P_1", 0, "POW_FAST_LOG_POLY", 6,
     "Degree 3 to 8 coefficients of `log(1 + z) - z + z^2/2` for the fast path."),
    ("pow/pow.h", "Q_1", 0, "POW_FAST_EXP_POLY", 5,
     "Degree 0 to 4 coefficients of `exp(z)` for the fast path."),
]

DINT_LAYOUT = "`(hi, lo, ex, sgn)` per entry, value `(-1)^sgn * (hi + lo / 2^64) * 2^(ex - 63)`"

# (C array name, Rust name, expected entry count, doc), from pow/dint.h
POW_DINT_TABLES = [
    ("_INVERSE_2_1", "POW_DINT_INVERSE_1", 92,
     f"Accurate-path first reciprocals of `i / 2^7`, `90 <= i <= 181`: {DINT_LAYOUT}."),
    ("_INVERSE_2_2", "POW_DINT_INVERSE_2", 129,
     f"Accurate-path second reciprocals of `j / 2^13`, `8128 <= j <= 8256`, same layout."),
    ("_LOG_INV_2_1", "POW_DINT_LOG_INV_1", 92,
     "`-log` of each `POW_DINT_INVERSE_1` entry, same layout."),
    ("_LOG_INV_2_2", "POW_DINT_LOG_INV_2", 129,
     "`-log` of each `POW_DINT_INVERSE_2` entry, same layout."),
    ("T1_2", "POW_DINT_T1", 64, "`2^(i/2^6)` for `0 <= i < 64`, same layout."),
    ("T2_2", "POW_DINT_T2", 64, "`2^(i/2^12)` for `0 <= i < 64`, same layout."),
    ("P_2", "POW_DINT_LOG_POLY", 9,
     "Accurate-path polynomial for `log(1 + z)`, highest degree first, same layout."),
    ("Q_2", "POW_DINT_EXP_POLY", 8,
     "Accurate-path polynomial for `exp(z)`, highest degree first, same layout."),
]

# (C scalar name, Rust name, doc), from pow/dint.h
POW_DINT_SCALARS = [
    ("LOG2_INV", "POW_DINT_LOG2_INV",
     "`2^12 / log(2)` in the dint layout, absolute error below `2^-52.96`."),
]

# (pow/dint.h scalar, log/dint.h scalar) pairs that must be equal
POW_DINT_SHARED = [("LOG2", "LOG2"), ("M_ONE", "M_ONE")]

QINT_LAYOUT = ("`(hh, hl, lh, ll, ex, sgn)` per entry, value "
               "`(-1)^sgn * (hh + hl / 2^64 + lh / 2^128 + ll / 2^192) * 2^(ex - 63)`")

# (C array name, Rust name, expected entry count, doc), from pow/qint.h
QINT_TABLES = [
    ("_LOG_INV_3_1", "POW_QINT_LOG_INV_1", 92,
     f"Extended-path `-log` of each `POW_DINT_INVERSE_1` entry: {QINT_LAYOUT}."),
    ("_LOG_INV_3_2", "POW_QINT_LOG_INV_2", 129,
     "Extended-path `-log` of each `POW_DINT_INVERSE_2` entry, same layout."),
    ("T1_3", "POW_QINT_T1", 64, "`2^(i/2^6)` for `0 <= i < 64`, same layout."),
    ("T2_3", "POW_QINT_T2", 64, "`2^(i/2^12)` for `0 <= i < 64`, same layout."),
    ("P_3", "POW_QINT_LOG_POLY", 18,
     "Extended-path polynomial for `log(1 + z)`, highest degree first, same layout."),
    ("Q_3", "POW_QINT_EXP_POLY", 15,
     "Extended-path polynomial for `exp(z)`, highest degree first, same layout."),
]

ACOS_SOURCE = "acos/acos.c"

# (C array name, occurrence index, Rust name, expected length, doc), from acos/acos.c.
# Occurrence 1 of `c`: occurrence 0 is the scalar `c` of the tiny path.
ACOS_TABLES = [
    ("cc", 0, "ACOS_CC", 264,
     "Row `j` at `8 * j`, `0 <= j <= 32`: `asin(x) = x * (c0 + c1 + t * P(t))` with "
     "`t = x^2 - j/128`, `c0 + c1` a double-double and `P` of degree 5 in `c2..c7`."),
    ("s", 0, "ACOS_S", 66,
     "`(lo, hi)` per entry: `sin(j * pi/64)` for `0 <= j <= 32`."),
    ("c", 1, "ACOS_C", 10,
     "`(hi, lo)` per entry: degree 1, 3, 5, 7 and 9 coefficients of `asin(v)` for the "
     "accurate path."),
    ("ct", 0, "ACOS_CT", 3,
     "Degree 11, 13 and 15 coefficients of `asin(v)` for the accurate path."),
]

ACOS_EXCEPTIONS_DOC = (
    "`acos` inputs whose accurate-path result lies too close to a rounding boundary: "
    "`(x, h, l)` per entry, in source order. The result is `h + l`.")
ACOS_EXCEPTION_COUNT = 5

QINT_SCALARS = [
    ("LOG2_Q", "POW_QINT_LN2", "`log(2)` in the qint layout, absolute error below `2^-256.14`."),
]

# (qint.h name, dint.h name, is_array): the qint value equals the dint value padded with zero limbs
QINT_FROM_DINT = [
    ("_INVERSE_3_1", "_INVERSE_2_1", True),
    ("_INVERSE_3_2", "_INVERSE_2_2", True),
    ("LOG2_INV_Q", "LOG2_INV", False),
    ("M_ONE_Q", "M_ONE", False),
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


def acos_exceptions(text):
    """`(x, h, l)` bit patterns of each `if(x== X) return H +- L;` line of as_acos_refine."""
    hex_float = r"0x[0-9a-fA-F]*\.?[0-9a-fA-F]*p[-+]?\d+"
    entry = re.compile(
        r"if\s*\(\s*x\s*==\s*(" + hex_float + r")\s*\)\s*return\s*(" + hex_float
        + r")\s*([-+])\s*(" + hex_float + r")\s*;")
    words = []
    for x, h, sign, l in entry.findall(text):
        words += [to_bits(x), to_bits(h), to_bits(("-" if sign == "-" else "") + l)]
    return words


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


def qint_entries(body):
    entry = re.compile(
        r"\.hh\s*=\s*(0x[0-9a-fA-F]+)\s*,\s*\.hl\s*=\s*(0x[0-9a-fA-F]+)\s*,"
        r"\s*\.lh\s*=\s*(0x[0-9a-fA-F]+)\s*,\s*\.ll\s*=\s*(0x[0-9a-fA-F]+)\s*,"
        r"\s*\.ex\s*=\s*(-?\d+)\s*,\s*\.sgn\s*=\s*(0x[0-9a-fA-F]+)")
    words = []
    for hh, hl, lh, ll, ex, sgn in entry.findall(body):
        words += [int(hh, 16), int(hl, 16), int(lh, 16), int(ll, 16),
                  int(ex) & 0xFFFF_FFFF_FFFF_FFFF, int(sgn, 16)]
    return words


def dint_as_qint(words):
    out = []
    for pos in range(0, len(words), 4):
        hi, lo, ex, sgn = words[pos:pos + 4]
        out += [hi, lo, 0, 0, ex, sgn]
    return out


def dint_body(text, name, is_array, kind="dint64_t"):
    marker = f"static const {kind} {name}" + ("[] = {" if is_array else " = {")
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
        "// src/binary64/log/log.c, src/binary64/log/dint.h, src/binary64/log1p/log1p.c,",
        "// src/binary64/pow/pow.h, src/binary64/pow/dint.h, src/binary64/pow/qint.h and",
        "// src/binary64/acos/acos.c.",
        "// Do not edit by hand: rerun the generator.",
        "//",
        "// acos.c: Copyright (c) 2024-2025 Alexei Sibidanov.",
        "// exp.c: Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.",
        "// log1p.c: Copyright (c) 2024-2026 Alexei Sibidanov <sibid@uvic.ca>.",
        "// log.c: Copyright (c) 2022-2026 INRIA and CERN. Authors: Paul Zimmermann and Tom Hubrecht.",
        "// log/dint.h: Copyright (c) 2022 CERN. Author: Tom Hubrecht.",
        "// pow/pow.h: Copyright (c) 2022-2025 CERN and Inria. Authors: Tom Hubrecht and Paul Zimmermann.",
        "// pow/dint.h, pow/qint.h: Copyright (c) 2022, 2023 CERN and Inria.",
        "// Authors: Tom Hubrecht and Paul Zimmermann.",
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
    with open(f"{base}/pow/dint.h") as handle:
        pow_dint = strip_comments(handle.read())
    for pow_name, log_name in POW_DINT_SHARED:
        if (dint_entries(dint_body(pow_dint, pow_name, False))
                != dint_entries(dint_body(dint, log_name, False))):
            raise SystemExit(f"pow/dint.h {pow_name} differs from log/dint.h {log_name}")
    for cname, rname, doc in POW_DINT_SCALARS:
        words = dint_entries(dint_body(pow_dint, cname, False))
        if len(words) != 4:
            raise SystemExit(f"{rname}: malformed scalar")
        emit(out, rname, words, doc, 4)
    for cname, rname, count, doc in POW_DINT_TABLES:
        words = dint_entries(dint_body(pow_dint, cname, True))
        if len(words) != 4 * count:
            raise SystemExit(f"{rname}: {len(words) // 4} entries, expected {count}")
        emit(out, rname, words, doc, 4)
    with open(f"{base}/pow/qint.h") as handle:
        qint = strip_comments(handle.read())
    for qname, dname, is_array in QINT_FROM_DINT:
        qwords = qint_entries(dint_body(qint, qname, is_array, "qint64_t"))
        dwords = dint_entries(dint_body(pow_dint, dname, is_array))
        if not qwords or qwords != dint_as_qint(dwords):
            raise SystemExit(f"pow/qint.h {qname} differs from pow/dint.h {dname}")
    for cname, rname, doc in QINT_SCALARS:
        words = qint_entries(dint_body(qint, cname, False, "qint64_t"))
        if len(words) != 6:
            raise SystemExit(f"{rname}: malformed scalar")
        emit(out, rname, words, doc, 6)
    for cname, rname, count, doc in QINT_TABLES:
        words = qint_entries(dint_body(qint, cname, True, "qint64_t"))
        if len(words) != 6 * count:
            raise SystemExit(f"{rname}: {len(words) // 6} entries, expected {count}")
        emit(out, rname, words, doc, 6)
    with open(f"{base}/{ACOS_SOURCE}") as handle:
        acos = strip_comments(handle.read())
    words = acos_exceptions(acos)
    if len(words) != 3 * ACOS_EXCEPTION_COUNT:
        raise SystemExit(f"ACOS_EXCEPTIONS: {len(words) // 3} entries, "
                         f"expected {ACOS_EXCEPTION_COUNT}")
    emit(out, "ACOS_EXCEPTIONS", words, ACOS_EXCEPTIONS_DOC, 3)
    for cname, occurrence, rname, length, doc in ACOS_TABLES:
        body = array_body(acos, cname, occurrence)
        words = [to_bits(tok) for tok in TOKEN.findall(body)]
        if len(words) != length:
            raise SystemExit(f"{rname}: {len(words)} values, expected {length}")
        emit(out, rname, words, doc, 2 if rname == "ACOS_S" else 4)
    with open(target, "w") as handle:
        handle.write("\n".join(out).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
