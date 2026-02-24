// Level-scheduled ILU(0) factorization kernel

struct Ilu0Params {
    level_size: u32,
    n: u32,
    diagonal_shift: f32,
    level_start: u32,
}

@group(0) @binding(0) var<storage, read> level_rows: array<i32>;
@group(0) @binding(1) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> col_indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> values: array<f32>;
@group(0) @binding(4) var<storage, read> diag_indices: array<i32>;
@group(0) @binding(5) var<uniform> params: Ilu0Params;

@compute @workgroup_size(256)
fn ilu0_level_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.level_size) {
        return;
    }

    let i = level_rows[params.level_start + tid];
    let row_start = row_ptrs[i];
    let row_end = row_ptrs[i + 1];

    // Process columns k < i (for L factor)
    for (var idx_ik = row_start; idx_ik < row_end; idx_ik = idx_ik + 1) {
        let k = col_indices[idx_ik];
        if (k >= i) {
            break;
        }

        // Get diagonal U[k,k]
        let diag_k = diag_indices[k];
        var diag_val = values[diag_k];

        // Handle zero pivot
        if (abs(diag_val) < 1e-15) {
            if (params.diagonal_shift > 0.0) {
                values[diag_k] = params.diagonal_shift;
                diag_val = params.diagonal_shift;
            }
        }

        // L[i,k] = A[i,k] / U[k,k]
        let l_ik = values[idx_ik] / diag_val;
        values[idx_ik] = l_ik;

        // Update row i for columns j > k
        let k_start = row_ptrs[k];
        let k_end = row_ptrs[k + 1];

        for (var idx_kj = k_start; idx_kj < k_end; idx_kj = idx_kj + 1) {
            let j = col_indices[idx_kj];
            if (j <= k) {
                continue;
            }

            // Find A[i,j] if it exists (zero fill-in constraint)
            for (var idx_ij = row_start; idx_ij < row_end; idx_ij = idx_ij + 1) {
                if (col_indices[idx_ij] == j) {
                    values[idx_ij] = values[idx_ij] - l_ik * values[idx_kj];
                    break;
                }
                if (col_indices[idx_ij] > j) {
                    break;
                }
            }
        }
    }
}
