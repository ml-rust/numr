//! Basic matrix operations: trace, diagonal, identity
//!
//! F32 only - WGSL doesn't support F64.

/// Basic operations shader: trace, diag, diagflat, identity
pub const BASIC_OPS_SHADER: &str = r#"
// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Trace - Sum of diagonal elements
// ============================================================================

struct TraceParams {
    n: u32,
    stride: u32,
}

@group(0) @binding(0) var<storage, read_write> trace_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> trace_output: array<f32>;
@group(0) @binding(2) var<uniform> trace_params: TraceParams;

var<workgroup> trace_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn trace_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>,
             @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = trace_params.n;
    let stride = trace_params.stride;

    // Each thread loads one diagonal element
    var val: f32 = 0.0;
    if (gid < n) {
        let idx = gid * stride + gid;
        val = trace_input[idx];
    }
    trace_shared[tid] = val;
    workgroupBarrier();

    // Parallel reduction
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            trace_shared[tid] = trace_shared[tid] + trace_shared[tid + s];
        }
        workgroupBarrier();
    }

    // First thread writes result (atomic add for multi-workgroup)
    if (tid == 0u) {
        trace_output[0] = trace_output[0] + trace_shared[0];
    }
}

// ============================================================================
// Diag - Extract diagonal elements
// ============================================================================

struct DiagParams {
    min_dim: u32,
    n_cols: u32,
}

@group(0) @binding(0) var<storage, read_write> diag_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> diag_output: array<f32>;
@group(0) @binding(2) var<uniform> diag_params: DiagParams;

@compute @workgroup_size(256)
fn diag_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let min_dim = diag_params.min_dim;
    let n_cols = diag_params.n_cols;

    if (gid < min_dim) {
        let idx = gid * n_cols + gid;
        diag_output[gid] = diag_input[idx];
    }
}

// ============================================================================
// Diagflat - Create diagonal matrix from vector
// ============================================================================

struct DiagflatParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> diagflat_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> diagflat_output: array<f32>;
@group(0) @binding(2) var<uniform> diagflat_params: DiagflatParams;

@compute @workgroup_size(256)
fn diagflat_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let n = diagflat_params.n;
    let total = n * n;

    if (gid < total) {
        let row = gid / n;
        let col = gid % n;
        if (row == col) {
            diagflat_output[gid] = diagflat_input[row];
        } else {
            diagflat_output[gid] = 0.0;
        }
    }
}

// ============================================================================
// Create Identity Matrix
// ============================================================================

struct IdentityParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> identity_output: array<f32>;
@group(0) @binding(1) var<uniform> identity_params: IdentityParams;

@compute @workgroup_size(256)
fn create_identity_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let n = identity_params.n;
    let total = n * n;

    if (gid < total) {
        let row = gid / n;
        let col = gid % n;
        if (row == col) {
            identity_output[gid] = 1.0;
        } else {
            identity_output[gid] = 0.0;
        }
    }
}

// ============================================================================
// Kronecker Product - A ⊗ B
// ============================================================================

struct KronParams {
    m_a: u32,
    n_a: u32,
    m_b: u32,
    n_b: u32,
}

@group(0) @binding(0) var<storage, read_write> kron_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> kron_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> kron_output: array<f32>;
@group(0) @binding(3) var<uniform> kron_params: KronParams;

@compute @workgroup_size(256)
fn kron_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let m_a = kron_params.m_a;
    let n_a = kron_params.n_a;
    let m_b = kron_params.m_b;
    let n_b = kron_params.n_b;

    let m_out = m_a * m_b;
    let n_out = n_a * n_b;
    let total = m_out * n_out;

    if (gid < total) {
        // Convert linear index to 2D coordinates in output
        let i_out = gid / n_out;
        let j_out = gid % n_out;

        // Find corresponding indices in A and B
        let i_a = i_out / m_b;
        let i_b = i_out % m_b;
        let j_a = j_out / n_b;
        let j_b = j_out % n_b;

        // Get values from A and B
        let a_val = kron_a[i_a * n_a + j_a];
        let b_val = kron_b[i_b * n_b + j_b];

        // out[i_a * m_b + i_b, j_a * n_b + j_b] = A[i_a, j_a] * B[i_b, j_b]
        kron_output[gid] = a_val * b_val;
    }
}
"#;
