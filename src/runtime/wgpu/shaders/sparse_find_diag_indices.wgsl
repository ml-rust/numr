// Find diagonal indices in CSR matrix

struct DiagParams {
    n: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> diag_indices: array<i32>;
@group(0) @binding(3) var<uniform> params: DiagParams;

@compute @workgroup_size(256)
fn find_diag_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = i32(gid.x);
    if (u32(row) >= params.n) {
        return;
    }

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];

    diag_indices[row] = -1;  // Default: no diagonal found

    for (var idx = start; idx < end; idx = idx + 1) {
        if (col_indices[idx] == row) {
            diag_indices[row] = idx;
            break;
        }
    }
}
