// Sparse format conversion kernels. Every index is i32. Values move as raw
// 32-bit words, so one module serves F32, I32 and U32.
//
// expand_ptrs:    compressed pointers -> one explicit index per entry
// histogram:      entries per bucket (row or column)
// scatter_order:  group entry ids by bucket, order inside a bucket arbitrary
// rank_gather:    order each bucket by (secondary index, entry id) and gather
//
// histogram + exclusive scan + scatter_order + rank_gather is a stable
// counting sort by (primary, secondary). The output matches a stable host
// sort exactly, whatever order the atomics in scatter_order produce.

// ============================================================================
// expand_ptrs
// ============================================================================

struct ExpandParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> ep_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read_write> ep_out: array<i32>;
@group(0) @binding(2) var<uniform> ep_params: ExpandParams;

@compute @workgroup_size(256)
fn expand_ptrs(
    @builtin(global_invocation_id) gid_grid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // Linear thread index over the 2-D dispatch grid.
    let bucket = gid_grid.x + gid_grid.y * nwg.x * 256u;
    if (bucket >= ep_params.n) {
        return;
    }
    let start = ep_ptrs[bucket];
    let end = ep_ptrs[bucket + 1u];
    for (var i = start; i < end; i = i + 1) {
        ep_out[i] = i32(bucket);
    }
}

// ============================================================================
// histogram
// ============================================================================

struct CountParams {
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> hist_keys: array<i32>;
@group(0) @binding(1) var<storage, read_write> hist_counts: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> hist_params: CountParams;

@compute @workgroup_size(256)
fn histogram(
    @builtin(global_invocation_id) gid_grid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid_grid.x + gid_grid.y * nwg.x * 256u;
    if (idx >= hist_params.nnz) {
        return;
    }
    atomicAdd(&hist_counts[u32(hist_keys[idx])], 1);
}

// ============================================================================
// scatter_order
// ============================================================================

@group(0) @binding(0) var<storage, read> so_keys: array<i32>;
@group(0) @binding(1) var<storage, read> so_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read_write> so_fill: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> so_order: array<i32>;
@group(0) @binding(4) var<uniform> so_params: CountParams;

@compute @workgroup_size(256)
fn scatter_order(
    @builtin(global_invocation_id) gid_grid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid_grid.x + gid_grid.y * nwg.x * 256u;
    if (idx >= so_params.nnz) {
        return;
    }
    let key = u32(so_keys[idx]);
    let slot = so_ptrs[key] + atomicAdd(&so_fill[key], 1);
    so_order[slot] = i32(idx);
}

// ============================================================================
// rank_gather
// ============================================================================

@group(0) @binding(0) var<storage, read> rg_primary: array<i32>;
@group(0) @binding(1) var<storage, read> rg_secondary: array<i32>;
@group(0) @binding(2) var<storage, read> rg_values: array<u32>;
@group(0) @binding(3) var<storage, read> rg_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> rg_order: array<i32>;
@group(0) @binding(5) var<storage, read_write> rg_out_secondary: array<i32>;
@group(0) @binding(6) var<storage, read_write> rg_out_values: array<u32>;
@group(0) @binding(7) var<uniform> rg_params: CountParams;

@compute @workgroup_size(256)
fn rank_gather(
    @builtin(global_invocation_id) gid_grid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let slot = gid_grid.x + gid_grid.y * nwg.x * 256u;
    if (slot >= rg_params.nnz) {
        return;
    }
    let entry = rg_order[slot];
    let key = u32(rg_primary[entry]);
    let secondary = rg_secondary[entry];
    let start = rg_ptrs[key];
    let end = rg_ptrs[key + 1u];

    // Rank of this entry inside its bucket by (secondary, entry id).
    var rank: i32 = 0;
    for (var m = start; m < end; m = m + 1) {
        let other = rg_order[m];
        let other_secondary = rg_secondary[other];
        if (other_secondary < secondary || (other_secondary == secondary && other < entry)) {
            rank = rank + 1;
        }
    }

    let dst = u32(start + rank);
    rg_out_secondary[dst] = secondary;
    rg_out_values[dst] = rg_values[entry];
}
