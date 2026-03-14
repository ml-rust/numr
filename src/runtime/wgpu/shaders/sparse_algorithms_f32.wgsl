// Sparse Algorithm Shaders - F32
//
// Column-Parallel Dense x Sparse Matrix Multiplication (DSMM)
// Sparse x Sparse Matrix Multiplication (SpGEMM) - symbolic, accumulate, scatter phases

// ============================================================================
// DSMM: C = A * B  (Dense A [M,K] x Sparse B CSC [K,N] -> Dense C [M,N])
// Each thread computes one element C[row, col]
// ============================================================================

struct DsmmParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> dsmm_a: array<f32>;
@group(0) @binding(1) var<storage, read> dsmm_col_ptrs: array<i32>;
@group(0) @binding(2) var<storage, read> dsmm_row_indices: array<i32>;
@group(0) @binding(3) var<storage, read> dsmm_b_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> dsmm_c: array<f32>;
@group(0) @binding(5) var<uniform> dsmm_params: DsmmParams;

@compute @workgroup_size(256)
fn dsmm_csc_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dsmm_params.m * dsmm_params.n;
    if (idx >= total) {
        return;
    }

    let row = idx / dsmm_params.n;
    let col = idx % dsmm_params.n;

    let col_start = dsmm_col_ptrs[col];
    let col_end = dsmm_col_ptrs[col + 1u];

    var sum: f32 = 0.0;
    for (var j: i32 = col_start; j < col_end; j = j + 1) {
        let k = dsmm_row_indices[j];
        let b_val = dsmm_b_values[j];
        let a_idx = row * dsmm_params.k + u32(k);
        sum = sum + dsmm_a[a_idx] * b_val;
    }

    dsmm_c[idx] = sum;
}

// ============================================================================
// SpGEMM Symbolic Phase: count NNZ per output row
// CSR A [M,K] x CSR B [K,N] -> row_nnz[M]
// Uses bitmap for small N
// ============================================================================

struct SymbolicParams {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> sym_a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> sym_a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> sym_b_row_ptrs: array<i32>;
@group(0) @binding(3) var<storage, read> sym_b_col_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> sym_row_nnz: array<i32>;
@group(0) @binding(5) var<storage, read_write> sym_bitmap: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> sym_params: SymbolicParams;

@compute @workgroup_size(256)
fn spgemm_symbolic_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= sym_params.m) {
        return;
    }

    let words_per_row = (sym_params.n + 31u) / 32u;
    let bitmap_offset = row * words_per_row;

    for (var w: u32 = 0u; w < words_per_row; w = w + 1u) {
        atomicStore(&sym_bitmap[bitmap_offset + w], 0u);
    }

    let a_start = sym_a_row_ptrs[row];
    let a_end = sym_a_row_ptrs[row + 1u];

    for (var ai: i32 = a_start; ai < a_end; ai = ai + 1) {
        let k = sym_a_col_indices[ai];

        let b_start = sym_b_row_ptrs[k];
        let b_end = sym_b_row_ptrs[k + 1];

        for (var bi: i32 = b_start; bi < b_end; bi = bi + 1) {
            let j = sym_b_col_indices[bi];
            let word_idx = bitmap_offset + u32(j) / 32u;
            let bit_idx = u32(j) % 32u;
            atomicOr(&sym_bitmap[word_idx], 1u << bit_idx);
        }
    }

    var count: i32 = 0;
    for (var w: u32 = 0u; w < words_per_row; w = w + 1u) {
        let word = atomicLoad(&sym_bitmap[bitmap_offset + w]);
        count = count + i32(countOneBits(word));
    }

    sym_row_nnz[row] = count;
}

// ============================================================================
// SpGEMM Accumulate Phase
// CSR A [M,K] x CSR B [K,N] -> dense row accumulators
// ============================================================================

struct SpgemmParams {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> accum_a_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> accum_a_col_indices: array<i32>;
@group(0) @binding(2) var<storage, read> accum_a_values: array<f32>;
@group(0) @binding(3) var<storage, read> accum_b_row_ptrs: array<i32>;
@group(0) @binding(4) var<storage, read> accum_b_col_indices: array<i32>;
@group(0) @binding(5) var<storage, read> accum_b_values: array<f32>;
@group(0) @binding(6) var<storage, read_write> accum_dense: array<f32>;
@group(0) @binding(7) var<storage, read_write> accum_flags: array<u32>;
@group(0) @binding(8) var<uniform> accum_params: SpgemmParams;

@compute @workgroup_size(256)
fn spgemm_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= accum_params.m) {
        return;
    }

    let accum_offset = row * accum_params.n;

    for (var col: u32 = 0u; col < accum_params.n; col = col + 1u) {
        accum_dense[accum_offset + col] = 0.0;
        accum_flags[accum_offset + col] = 0u;
    }

    let a_start = accum_a_row_ptrs[row];
    let a_end = accum_a_row_ptrs[row + 1u];

    for (var ai: i32 = a_start; ai < a_end; ai = ai + 1) {
        let k = accum_a_col_indices[ai];
        let a_val = accum_a_values[ai];

        let b_start = accum_b_row_ptrs[k];
        let b_end = accum_b_row_ptrs[k + 1];

        for (var bi: i32 = b_start; bi < b_end; bi = bi + 1) {
            let j = accum_b_col_indices[bi];
            let b_val = accum_b_values[bi];
            let idx = accum_offset + u32(j);
            accum_dense[idx] = accum_dense[idx] + a_val * b_val;
            accum_flags[idx] = 1u;
        }
    }
}

// ============================================================================
// SpGEMM Scatter Phase
// Compact dense row accumulators into CSR output arrays
// ============================================================================

@group(0) @binding(0) var<storage, read> scatter_c_row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> scatter_accum: array<f32>;
@group(0) @binding(2) var<storage, read> scatter_flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> scatter_c_col_indices: array<i32>;
@group(0) @binding(4) var<storage, read_write> scatter_c_values: array<f32>;
@group(0) @binding(5) var<uniform> scatter_params: SpgemmParams;

@compute @workgroup_size(256)
fn spgemm_scatter_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= scatter_params.m) {
        return;
    }

    let accum_offset = row * scatter_params.n;
    var write_idx: i32 = scatter_c_row_ptrs[row];

    for (var col: u32 = 0u; col < scatter_params.n; col = col + 1u) {
        let idx = accum_offset + col;
        if (scatter_flags[idx] != 0u) {
            scatter_c_col_indices[write_idx] = i32(col);
            scatter_c_values[write_idx] = scatter_accum[idx];
            write_idx = write_idx + 1;
        }
    }
}
