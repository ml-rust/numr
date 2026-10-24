// Shared prelude for the casts between I64 and the 32-bit dtypes.
//
// WGSL has no 64-bit integer. An I64 buffer binds as `array<vec2<u32>>`:
// `x` is the low word and `y` the high word, matching the little-endian
// byte order of the tensor storage. Every 32-bit buffer binds as
// `array<u32>`, so the kernels move bit patterns and no float read or write
// can flush or canonicalize a value.
//
// The semantics follow the CPU reference (`cast_kernel` in
// runtime/cpu/kernels/memory.rs): it converts the source to f64 and then
// applies a Rust `as` cast to the target.
//
// A dispatch dimension holds at most 65535 workgroups, so threads spread
// over a 2-D grid.

const WORKGROUP_SIZE: u32 = 256u;
const GRID_X: u32 = 65535u;

struct CastParams {
    numel: u32,
}

fn cast_thread_index(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * GRID_X * WORKGROUP_SIZE;
}

// Number of significant bits in an unsigned 64-bit value. Zero has none.
fn u64_bit_len(v: vec2<u32>) -> u32 {
    if (v.y != 0u) {
        return 64u - countLeadingZeros(v.y);
    }
    return 32u - countLeadingZeros(v.x);
}

// A runtime WGSL shift takes its amount modulo 32, so every case at or past
// 32 bits is spelled out.
fn u64_shl(v: vec2<u32>, s: u32) -> vec2<u32> {
    if (s == 0u) {
        return v;
    }
    if (s >= 64u) {
        return vec2<u32>(0u, 0u);
    }
    if (s >= 32u) {
        return vec2<u32>(0u, v.x << (s - 32u));
    }
    return vec2<u32>(v.x << s, (v.y << s) | (v.x >> (32u - s)));
}

fn u64_shr(v: vec2<u32>, s: u32) -> vec2<u32> {
    if (s == 0u) {
        return v;
    }
    if (s >= 64u) {
        return vec2<u32>(0u, 0u);
    }
    if (s >= 32u) {
        return vec2<u32>(v.y >> (s - 32u), 0u);
    }
    return vec2<u32>((v.x >> s) | (v.y << (32u - s)), v.y >> s);
}

fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var borrow = 0u;
    if (a.x < b.x) {
        borrow = 1u;
    }
    return vec2<u32>(a.x - b.x, a.y - b.y - borrow);
}

fn u64_add_one(v: vec2<u32>) -> vec2<u32> {
    let lo = v.x + 1u;
    var hi = v.y;
    if (lo == 0u) {
        hi = hi + 1u;
    }
    return vec2<u32>(lo, hi);
}

// Two's-complement negation: `~v + 1`. The carry reaches the high word only
// when the low word is zero.
fn u64_neg(v: vec2<u32>) -> vec2<u32> {
    let lo = ~v.x + 1u;
    var hi = ~v.y;
    if (lo == 0u) {
        hi = hi + 1u;
    }
    return vec2<u32>(lo, hi);
}

// Round `v` to `keep` significant bits, ties to even. The result keeps its
// scale: it is `v` with the dropped low bits rounded away. The callers pass
// magnitudes of at most 2^63, so rounding up never overflows 64 bits.
fn u64_round_to_bits(v: vec2<u32>, keep: u32) -> vec2<u32> {
    let n = u64_bit_len(v);
    if (n <= keep) {
        return v;
    }
    let k = n - keep;
    var q = u64_shr(v, k);
    let rem = u64_sub(v, u64_shl(q, k));
    let halfway = u64_shl(vec2<u32>(1u, 0u), k - 1u);
    let tie = rem.x == halfway.x && rem.y == halfway.y;
    if (u64_lt(halfway, rem) || (tie && (q.x & 1u) == 1u)) {
        q = u64_add_one(q);
    }
    return u64_shl(q, k);
}

// f32 bits of a nonzero magnitude with at most 24 significant bits. Such a
// value is an exact f32 normal: its exponent is `bit_len - 1` and its
// significand is the top 24 bits.
fn u64_to_f32_bits(v: vec2<u32>) -> u32 {
    let n = u64_bit_len(v);
    var m: u32;
    if (n > 24u) {
        m = u64_shr(v, n - 24u).x;
    } else {
        m = v.x << (24u - n);
    }
    return ((n - 1u + 127u) << 23u) | (m & 0x7fffffu);
}
