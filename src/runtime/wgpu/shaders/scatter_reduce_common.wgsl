// Shared front half of the scatter_reduce reduce shaders.
//
// Concatenated BEFORE scatter_reduce_{f32,i32,u32}.wgsl. WGSL has no include
// and no forward declarations, so that order is load-bearing.
//
// One invocation per destination element finds its run in the sorted keys by
// binary search and folds the run in order. The run lists source positions in
// increasing order, the order the CPU reference `scatter_reduce_kernel` in
// src/runtime/cpu/kernels/index/scatter.rs combines them in, so the result is
// bit-identical run to run and to CPU.
//
// Bindings: 0 src, 1 sorted keys, 2 sorted source positions, 3 destination,
// 4 output, 5 params. The dtype file declares 0, 3 and 4.

const SR_THREADS: u32 = 256u;

const SR_SUM: u32 = 0u;
const SR_PROD: u32 = 1u;
const SR_MAX: u32 = 2u;
const SR_MIN: u32 = 3u;
const SR_MEAN: u32 = 4u;

struct SrReduceParams {
    n: u32,
    dst_numel: u32,
    include_self: u32,
    _pad: u32,
}

@group(0) @binding(1) var<storage, read> sr_keys: array<u32>;
@group(0) @binding(2) var<storage, read> sr_vals: array<u32>;
@group(0) @binding(5) var<uniform> sr_params: SrReduceParams;

fn sr_flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * SR_THREADS;
}

// First position in the sorted keys whose key is not below `want`.
fn sr_lower_bound(want: u32) -> u32 {
    var lo = 0u;
    var hi = sr_params.n;
    while (lo < hi) {
        let mid = lo + ((hi - lo) >> 1u);
        if (sr_keys[mid] < want) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Mean's denominator: the run length, plus the destination's own value when
// include_self is set, as the CPU kernel's count seed of 1 does.
fn sr_mean_count(lo: u32, hi: u32) -> u32 {
    return (hi - lo) + sr_params.include_self;
}
