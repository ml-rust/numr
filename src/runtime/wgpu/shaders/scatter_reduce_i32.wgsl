// scatter_reduce reductions for i32.
//
// Concatenated after int_saturate.wgsl, int_matmul_acc.wgsl, int_wide_div.wgsl
// and scatter_reduce_common.wgsl, in that order. WGSL has no include and no
// forward declarations, so the order is load-bearing.
//
// An integer reduction accumulates, and accumulators saturate rather than wrap
// (runtime/cpu/kernels/wide_acc.rs). CPU runs the total in i128 and narrows
// once (runtime/cpu/kernels/scatter_reduce_int.rs). Here:
//
//  * sum and mean run in a 64-bit accumulator. A run holds fewer than 2^32
//    elements, so the exact total fits, and the one narrow saturates as
//    CPU's does. `mean` divides once, truncating, before the narrow.
//  * prod keeps magnitude plus sign parity: a factor of 0 pins the product at
//    0, and every other factor has magnitude at least 1, so a magnitude that
//    has left the range never comes back. That state is exact for every
//    representable product and clamps to the correctly signed bound otherwise.
//  * max and min compare, which is exact in the element type.

@group(0) @binding(0) var<storage, read> sr_src: array<i32>;
@group(0) @binding(3) var<storage, read> sr_dst: array<i32>;
@group(0) @binding(4) var<storage, read_write> sr_out: array<i32>;

// Seed of a destination element: its own value when include_self is set,
// otherwise the reduction's identity. Max and min seed from the type's bounds,
// which is where CPU's `from_f64(-inf)` and `from_f64(inf)` saturate.
fn sr_seed(d: u32, op: u32) -> i32 {
    if (sr_params.include_self != 0u) {
        return sr_dst[d];
    }
    if (op == SR_PROD) {
        return 1;
    }
    if (op == SR_MAX) {
        return NUMR_I32_MIN;
    }
    if (op == SR_MIN) {
        return NUMR_I32_MAX;
    }
    return 0;
}

fn sr_reduce(d: u32, op: u32) {
    let seed = sr_seed(d, op);
    let lo = sr_lower_bound(d);
    let hi = sr_lower_bound(d + 1u);

    if (op == SR_MAX || op == SR_MIN) {
        var best = seed;
        for (var j = lo; j < hi; j = j + 1u) {
            let v = sr_src[sr_vals[j]];
            if ((op == SR_MAX && v > best) || (op == SR_MIN && v < best)) {
                best = v;
            }
        }
        sr_out[d] = best;
        return;
    }

    if (op == SR_PROD) {
        var zero_seen = seed == 0;
        var negative = seed < 0;
        var saturated = false;
        var mag = numr_i32_magnitude(seed);
        for (var j = lo; j < hi; j = j + 1u) {
            let v = sr_src[sr_vals[j]];
            if (v == 0) {
                zero_seen = true;
            }
            negative = select(negative, !negative, v < 0);
            let v_mag = numr_i32_magnitude(v);
            if (!saturated) {
                if (numr_u32_mul_overflows(mag, v_mag)) {
                    saturated = true;
                } else {
                    mag = mag * v_mag;
                }
            }
        }
        sr_out[d] = numr_i32_product(zero_seen, saturated, negative, mag);
        return;
    }

    var acc = numr_i64_from_i32(seed);
    for (var j = lo; j < hi; j = j + 1u) {
        acc = numr_i64_add(acc, numr_i64_from_i32(sr_src[sr_vals[j]]));
    }
    if (op == SR_MEAN) {
        let count = sr_mean_count(lo, hi);
        if (count > 0u) {
            acc = numr_i64_div_u32_trunc(acc, count);
        }
    }
    sr_out[d] = numr_i64_to_i32_sat(acc);
}

@compute @workgroup_size(256)
fn scatter_reduce_sum_i32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_SUM);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_prod_i32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_PROD);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_max_i32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MAX);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_min_i32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MIN);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_mean_i32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MEAN);
    }
}
