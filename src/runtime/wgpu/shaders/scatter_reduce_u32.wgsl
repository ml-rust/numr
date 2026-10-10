// scatter_reduce reductions for u32.
//
// Concatenated after int_saturate.wgsl, int_matmul_acc.wgsl, int_wide_div.wgsl
// and scatter_reduce_common.wgsl, in that order. The signed twin,
// scatter_reduce_i32.wgsl, carries the reasoning; the unsigned case is the same
// without a sign to track. The 64-bit accumulator is unsigned here: fewer than
// 2^32 contributions below 2^32 each cannot reach 2^64.

@group(0) @binding(0) var<storage, read> sr_src: array<u32>;
@group(0) @binding(3) var<storage, read> sr_dst: array<u32>;
@group(0) @binding(4) var<storage, read_write> sr_out: array<u32>;

// Seed of a destination element: its own value when include_self is set,
// otherwise the reduction's identity. Max and min seed from the type's bounds,
// which is where CPU's `from_f64(-inf)` and `from_f64(inf)` saturate.
fn sr_seed(d: u32, op: u32) -> u32 {
    if (sr_params.include_self != 0u) {
        return sr_dst[d];
    }
    if (op == SR_PROD) {
        return 1u;
    }
    if (op == SR_MIN) {
        return NUMR_U32_MAX;
    }
    return 0u;
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
        var zero_seen = seed == 0u;
        var saturated = false;
        var mag = seed;
        for (var j = lo; j < hi; j = j + 1u) {
            let v = sr_src[sr_vals[j]];
            if (v == 0u) {
                zero_seen = true;
            }
            if (!saturated) {
                if (numr_u32_mul_overflows(mag, v)) {
                    saturated = true;
                } else {
                    mag = mag * v;
                }
            }
        }
        sr_out[d] = numr_u32_product(zero_seen, saturated, mag);
        return;
    }

    var acc = numr_u64_from_u32(seed);
    for (var j = lo; j < hi; j = j + 1u) {
        acc = numr_i64_add(acc, numr_u64_from_u32(sr_src[sr_vals[j]]));
    }
    if (op == SR_MEAN) {
        let count = sr_mean_count(lo, hi);
        if (count > 0u) {
            acc = numr_u64_div_u32(acc, count);
        }
    }
    sr_out[d] = numr_u64_to_u32_sat(acc);
}

@compute @workgroup_size(256)
fn scatter_reduce_sum_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_SUM);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_prod_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_PROD);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_max_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MAX);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_min_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MIN);
    }
}

@compute @workgroup_size(256)
fn scatter_reduce_mean_u32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = sr_flat_id(gid, nwg);
    if (d < sr_params.dst_numel) {
        sr_reduce(d, SR_MEAN);
    }
}
