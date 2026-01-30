//! Triangular system solvers: forward and backward substitution
//!
//! F32 only - WGSL doesn't support F64.

/// Triangular solvers shader: forward_sub, backward_sub
#[allow(dead_code)]
pub const SOLVERS_SHADER: &str = r#"
// ============================================================================
// Forward Substitution - Solve Lx = b
// ============================================================================

struct ForwardSubParams {
    n: u32,
    unit_diagonal: u32,
}

@group(0) @binding(0) var<storage, read_write> forward_l: array<f32>;
@group(0) @binding(1) var<storage, read_write> forward_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> forward_x: array<f32>;
@group(0) @binding(3) var<uniform> forward_params: ForwardSubParams;

@compute @workgroup_size(1)
fn forward_sub_f32() {
    let n = forward_params.n;
    let unit_diag = forward_params.unit_diagonal != 0u;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        var sum: f32 = forward_b[i];

        for (var j: u32 = 0u; j < i; j = j + 1u) {
            let l_idx = i * n + j;
            sum = sum - forward_l[l_idx] * forward_x[j];
        }

        if (unit_diag) {
            forward_x[i] = sum;
        } else {
            let diag_idx = i * n + i;
            forward_x[i] = sum / forward_l[diag_idx];
        }
    }
}

// ============================================================================
// Backward Substitution - Solve Ux = b
// ============================================================================

struct BackwardSubParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> backward_u: array<f32>;
@group(0) @binding(1) var<storage, read_write> backward_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> backward_x: array<f32>;
@group(0) @binding(3) var<uniform> backward_params: BackwardSubParams;

@compute @workgroup_size(1)
fn backward_sub_f32() {
    let n = backward_params.n;

    // Start from last row
    for (var ii: u32 = 0u; ii < n; ii = ii + 1u) {
        let i = n - 1u - ii;

        var sum: f32 = backward_b[i];

        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            let u_idx = i * n + j;
            sum = sum - backward_u[u_idx] * backward_x[j];
        }

        let diag_idx = i * n + i;
        backward_x[i] = sum / backward_u[diag_idx];
    }
}
"#;
