// Keys for scatter_reduce's grouping sort.
//
// One invocation per source element writes the flat destination position (the
// key) and the source's own flat position (the value). An out-of-range index
// gets key `invalid_key`, the destination element count, which sorts past every
// real destination and is never reduced. scatter_reduce_radix.wgsl then sorts
// the keys.
//
// The integer divisions below sit in straight-line code, not in a loop: the
// NVIDIA shader compiler fails with "NVVM compilation failed" on an integer
// divide inside a WGSL loop (see int_saturate.wgsl).

const SR_THREADS: u32 = 256u;

// Axes are right-aligned into 8 slots: an unused leading slot has extent 1 and
// stride 0, so walking all 8 slots leaves the key unchanged there.
struct SrKeyParams {
    n: u32,
    dim_slot: u32,
    dim_size: u32,
    invalid_key: u32,
    src_extent: array<vec4<u32>, 2>,
    dst_stride: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read> sr_key_index_i32: array<i32>;
@group(0) @binding(0) var<storage, read> sr_key_index_i64: array<u32>;
@group(0) @binding(1) var<storage, read_write> sr_key_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> sr_key_vals: array<u32>;
@group(0) @binding(3) var<uniform> sr_key_params: SrKeyParams;

fn sr_slot_extent(slot: u32) -> u32 {
    return sr_key_params.src_extent[slot >> 2u][slot & 3u];
}

fn sr_slot_stride(slot: u32) -> u32 {
    return sr_key_params.dst_stride[slot >> 2u][slot & 3u];
}

// Peels slot `slot`'s coordinate off `rem` and adds its destination offset.
fn sr_key_axis(slot: u32, index_val: u32, rem: ptr<function, u32>, offset: ptr<function, u32>) {
    let extent = sr_slot_extent(slot);
    var coord = *rem % extent;
    *rem = *rem / extent;
    if (slot == sr_key_params.dim_slot) {
        coord = index_val;
    }
    *offset = *offset + coord * sr_slot_stride(slot);
}

// Source position `i` decomposes over the source extents, innermost slot
// first. The destination position replaces the coordinate on the scatter axis
// with the index value and recombines with the destination strides, so a
// source shorter than the destination on another axis lands where the CPU
// reference puts it.
fn sr_key_offset(i: u32, index_val: u32) -> u32 {
    var rem = i;
    var offset = 0u;
    sr_key_axis(7u, index_val, &rem, &offset);
    sr_key_axis(6u, index_val, &rem, &offset);
    sr_key_axis(5u, index_val, &rem, &offset);
    sr_key_axis(4u, index_val, &rem, &offset);
    sr_key_axis(3u, index_val, &rem, &offset);
    sr_key_axis(2u, index_val, &rem, &offset);
    sr_key_axis(1u, index_val, &rem, &offset);
    sr_key_axis(0u, index_val, &rem, &offset);
    return offset;
}

fn sr_flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * SR_THREADS;
}

@compute @workgroup_size(256)
fn scatter_reduce_keys_i32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = sr_flat_id(gid, nwg);
    if (i >= sr_key_params.n) {
        return;
    }
    var key = sr_key_params.invalid_key;
    let index_val = sr_key_index_i32[i];
    if (index_val >= 0 && u32(index_val) < sr_key_params.dim_size) {
        key = sr_key_offset(i, u32(index_val));
    }
    sr_key_keys[i] = key;
    sr_key_vals[i] = i;
}

// I64 indices arrive as two u32 words, low word first. An index is in range
// only when its high word is zero, which also rejects every negative index.
@compute @workgroup_size(256)
fn scatter_reduce_keys_i64(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = sr_flat_id(gid, nwg);
    if (i >= sr_key_params.n) {
        return;
    }
    var key = sr_key_params.invalid_key;
    let lo = sr_key_index_i64[i * 2u];
    let hi = sr_key_index_i64[i * 2u + 1u];
    if (hi == 0u && lo < sr_key_params.dim_size) {
        key = sr_key_offset(i, lo);
    }
    sr_key_keys[i] = key;
    sr_key_vals[i] = i;
}
