// Level-scheduled IC(0) factorization kernel

struct Ic0Params {
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
@group(0) @binding(5) var<uniform> params: Ic0Params;

@compute @workgroup_size(256)
fn ic0_level_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.level_size) {
        return;
    }

    let i = level_rows[params.level_start + tid];
    let i_start = row_ptrs[i];
    let i_end = row_ptrs[i + 1];

    // Process off-diagonal entries in row i (columns k < i)
    for (var idx_ik = i_start; idx_ik < i_end; idx_ik = idx_ik + 1) {
        let k = col_indices[idx_ik];
        if (k >= i) {
            break;
        }

        let k_start = row_ptrs[k];
        let k_end = row_ptrs[k + 1];

        // Compute inner product contribution
        var sum = values[idx_ik];

        for (var idx_kj = k_start; idx_kj < k_end; idx_kj = idx_kj + 1) {
            let j = col_indices[idx_kj];
            if (j >= k) {
                break;
            }

            // Check if L[i,j] exists
            for (var idx_ij = i_start; idx_ij < i_end; idx_ij = idx_ij + 1) {
                if (col_indices[idx_ij] == j) {
                    sum = sum - values[idx_ij] * values[idx_kj];
                    break;
                }
                if (col_indices[idx_ij] > j) {
                    break;
                }
            }
        }

        // Divide by L[k,k]
        let diag_k = diag_indices[k];
        values[idx_ik] = sum / values[diag_k];
    }

    // Compute diagonal L[i,i]
    let diag_i = diag_indices[i];
    var diag_sum = values[diag_i] + params.diagonal_shift;

    for (var idx_ij = i_start; idx_ij < i_end; idx_ij = idx_ij + 1) {
        let j = col_indices[idx_ij];
        if (j >= i) {
            break;
        }
        diag_sum = diag_sum - values[idx_ij] * values[idx_ij];
    }

    if (diag_sum <= 0.0) {
        diag_sum = select(1e-10, params.diagonal_shift, params.diagonal_shift > 0.0);
    }

    values[diag_i] = sqrt(diag_sum);
}
