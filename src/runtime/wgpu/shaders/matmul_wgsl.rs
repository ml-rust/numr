//! WGSL shader source code for matrix multiplication
//!
//! Implements tiled matrix multiplication for better memory access patterns.
//! Supports 2D and batched matrix multiplication.

/// Matrix multiplication shader module source (F32 only)
pub const MATMUL_SHADER: &str = r#"
// ============================================================================
// Workgroup Configuration
// ============================================================================

// Tile size for tiled matrix multiplication
const TILE_SIZE: u32 = 16u;

// Shared memory for tile loading
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

// ============================================================================
// Matrix Multiplication Parameters
// ============================================================================

struct MatmulParams {
    M: u32,             // Rows of A and C
    K: u32,             // Cols of A, Rows of B
    N: u32,             // Cols of B and C
    batch_size: u32,    // Number of matrices in batch (1 for non-batched)
}

@group(0) @binding(0) var<storage, read_write> matmul_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> matmul_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matmul_c: array<f32>;
@group(0) @binding(3) var<uniform> matmul_params: MatmulParams;

// ============================================================================
// 2D Matrix Multiplication (C = A @ B)
// ============================================================================

// Uses tiled algorithm with shared memory for better cache utilization.
// Each workgroup computes a TILE_SIZE x TILE_SIZE tile of the output.

@compute @workgroup_size(16, 16, 1)
fn matmul_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    var sum: f32 = 0.0;

    // Number of tiles along K dimension
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile from A
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_a[local_id.y][local_id.x] = matmul_a[row * K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }

        // Load tile from B
        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_b[local_id.y][local_id.x] = matmul_b[b_row * N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < M && col < N) {
        matmul_c[row * N + col] = sum;
    }
}

// ============================================================================
// Batched Matrix Multiplication (C[b] = A[b] @ B[b])
// ============================================================================

@compute @workgroup_size(16, 16, 1)
fn batched_matmul_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                      @builtin(local_invocation_id) local_id: vec3<u32>,
                      @builtin(workgroup_id) group_id: vec3<u32>) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;
    let batch_size = matmul_params.batch_size;

    let batch = group_id.z;
    if (batch >= batch_size) {
        return;
    }

    let row = group_id.y * TILE_SIZE + local_id.y;
    let col = group_id.x * TILE_SIZE + local_id.x;

    // Batch offsets
    let a_batch_offset = batch * M * K;
    let b_batch_offset = batch * K * N;
    let c_batch_offset = batch * M * N;

    var sum: f32 = 0.0;

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile from A
        let a_col = t * TILE_SIZE + local_id.x;
        if (row < M && a_col < K) {
            tile_a[local_id.y][local_id.x] = matmul_a[a_batch_offset + row * K + a_col];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }

        // Load tile from B
        let b_row = t * TILE_SIZE + local_id.y;
        if (b_row < K && col < N) {
            tile_b[local_id.y][local_id.x] = matmul_b[b_batch_offset + b_row * N + col];
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        matmul_c[c_batch_offset + row * N + col] = sum;
    }
}

// ============================================================================
// Simple (non-tiled) Matrix Multiplication for small matrices
// ============================================================================

// For very small matrices, tiling overhead may hurt performance.
// This version is simpler and may be faster for small matrices.

@compute @workgroup_size(256, 1, 1)
fn matmul_simple_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = matmul_params.M;
    let K = matmul_params.K;
    let N = matmul_params.N;

    let idx = global_id.x;
    let total = M * N;

    if (idx >= total) {
        return;
    }

    let row = idx / N;
    let col = idx % N;

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        sum = sum + matmul_a[row * K + k] * matmul_b[k * N + col];
    }

    matmul_c[idx] = sum;
}

// ============================================================================
// Matrix-Vector Multiplication (y = A @ x)
// ============================================================================

struct MatvecParams {
    M: u32,     // Rows of A (and y)
    N: u32,     // Cols of A (and length of x)
}

@group(0) @binding(0) var<storage, read_write> matvec_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> matvec_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> matvec_y: array<f32>;
@group(0) @binding(3) var<uniform> matvec_params: MatvecParams;

var<workgroup> matvec_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn matvec_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let row = group_id.x;

    if (row >= matvec_params.M) {
        return;
    }

    let N = matvec_params.N;
    let row_offset = row * N;

    // Each thread computes partial dot product
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < N) {
        sum = sum + matvec_a[row_offset + i] * matvec_x[i];
        i = i + 256u;
    }

    matvec_shared[tid] = sum;
    workgroupBarrier();

    // Parallel reduction
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            matvec_shared[tid] = matvec_shared[tid] + matvec_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        matvec_y[row] = matvec_shared[0];
    }
}
"#;
