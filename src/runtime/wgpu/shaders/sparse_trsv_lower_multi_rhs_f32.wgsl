// Multi-RHS level-scheduled sparse lower triangular solve (forward substitution)
// Processes all (row, rhs_column) pairs in a single level in parallel

struct TrsvMultiRhsParams {
    level_size: u32,
    nrhs: u32,
    n: u32,
    unit_diagonal: u32,
    level_start: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> level_rows: array<i32>;
@group(0) @binding(1) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
@group(0) @binding(4) var<storage, read> b: array<f32>;
@group(0) @binding(5) var<storage, read_write> x: array<f32>;
@group(0) @binding(6) var<uniform> params: TrsvMultiRhsParams;

@compute @workgroup_size(256)
fn sparse_trsv_lower_level_multi_rhs_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_work = params.level_size * params.nrhs;
    if (tid >= total_work) {
        return;
    }

    let row_idx = tid / params.nrhs;
    let rhs_col = tid % params.nrhs;
    let row = level_rows[params.level_start + row_idx];

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    var sum = b[u32(row) * params.nrhs + rhs_col];
    var diag = f32(1.0);

    for (var idx = start; idx < end; idx = idx + 1) {
        let col = col_indices[idx];
        if (col < row) {
            sum = sum - values[idx] * x[u32(col) * params.nrhs + rhs_col];
        } else if (col == row && params.unit_diagonal == 0u) {
            diag = values[idx];
        }
    }

    if (params.unit_diagonal == 0u) {
        sum = sum / diag;
    }

    x[u32(row) * params.nrhs + rhs_col] = sum;
}
