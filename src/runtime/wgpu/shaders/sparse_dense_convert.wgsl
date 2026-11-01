// Sparse <-> dense conversion kernels. Every index is i32. Values move as raw
// 32-bit words, so one module serves F32, I32 and U32.
//
// csr_to_dense:      one thread per row writes the row's entries (last wins)
// dense_row_count:   kept entries per row
// dense_row_scatter: one thread per row writes its kept entries in column order
//
// The dense -> COO pair emits entries in row-major order, the same order as a
// host scan of the matrix.

// ============================================================================
// csr_to_dense
// ============================================================================

struct ShapeParams {
    nrows: u32,
    ncols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> ctd_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> ctd_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> ctd_values: array<u32>;
@group(0) @binding(3) var<storage, read_write> ctd_dense: array<u32>;
@group(0) @binding(4) var<uniform> ctd_params: ShapeParams;

@compute @workgroup_size(256)
fn csr_to_dense(
    @builtin(global_invocation_id) gid_grid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // Linear thread index over the 2-D dispatch grid.
    let row = gid_grid.x + gid_grid.y * nwg.x * 256u;
    if (row >= ctd_params.nrows) {
        return;
    }
    let start = ctd_row_ptrs[row];
    let end = ctd_row_ptrs[row + 1u];
    for (var i = start; i < end; i = i + 1) {
        let col = u32(ctd_col_indices[i]);
        ctd_dense[row * ctd_params.ncols + col] = ctd_values[i];
    }
}

// ============================================================================
// keep predicate: |value| >= threshold
// ============================================================================

// mode 0: f32, `threshold` holds the f32 bits of the smallest f32 >= the
//         host threshold.
// mode 1: i32, `threshold` holds the smallest kept magnitude as u32.
// mode 2: u32, `threshold` holds the smallest kept value.
// keep_none = 1 keeps no entry (NaN threshold, or one above every value).
struct KeepParams {
    nrows: u32,
    ncols: u32,
    threshold: u32,
    mode: u32,
    keep_none: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn keep_value(bits: u32, p: KeepParams) -> bool {
    if (p.keep_none != 0u) {
        return false;
    }
    if (p.mode == 0u) {
        // NaN compares false, as on the host.
        return abs(bitcast<f32>(bits)) >= bitcast<f32>(p.threshold);
    }
    if (p.mode == 1u) {
        // u32(i32) keeps the bits, so |i32::MIN| reads as 2^31.
        return u32(abs(bitcast<i32>(bits))) >= p.threshold;
    }
    return bits >= p.threshold;
}

// ============================================================================
// dense_row_count
// ============================================================================

@group(0) @binding(0) var<storage, read> drc_dense: array<u32>;
@group(0) @binding(1) var<storage, read_write> drc_counts: array<i32>;
@group(0) @binding(2) var<uniform> drc_params: KeepParams;

@compute @workgroup_size(256)
fn dense_row_count(
    @builtin(global_invocation_id) gid_grid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = gid_grid.x + gid_grid.y * nwg.x * 256u;
    if (row >= drc_params.nrows) {
        return;
    }
    let p = drc_params;
    var count: i32 = 0;
    for (var col: u32 = 0u; col < p.ncols; col = col + 1u) {
        if (keep_value(drc_dense[row * p.ncols + col], p)) {
            count = count + 1;
        }
    }
    drc_counts[row] = count;
}

// ============================================================================
// dense_row_scatter
// ============================================================================

@group(0) @binding(0) var<storage, read> drs_dense: array<u32>;
@group(0) @binding(1) var<storage, read> drs_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read_write> drs_rows: array<i32>;
@group(0) @binding(3) var<storage, read_write> drs_cols: array<i32>;
@group(0) @binding(4) var<storage, read_write> drs_values: array<u32>;
@group(0) @binding(5) var<uniform> drs_params: KeepParams;

@compute @workgroup_size(256)
fn dense_row_scatter(
    @builtin(global_invocation_id) gid_grid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = gid_grid.x + gid_grid.y * nwg.x * 256u;
    if (row >= drs_params.nrows) {
        return;
    }
    let p = drs_params;
    var dst = drs_ptrs[row];
    for (var col: u32 = 0u; col < p.ncols; col = col + 1u) {
        let bits = drs_dense[row * p.ncols + col];
        if (keep_value(bits, p)) {
            drs_rows[dst] = i32(row);
            drs_cols[dst] = i32(col);
            drs_values[dst] = bits;
            dst = dst + 1;
        }
    }
}
