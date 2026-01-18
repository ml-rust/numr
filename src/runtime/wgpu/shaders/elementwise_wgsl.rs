//! WGSL shader source code for element-wise operations
//!
//! Includes binary, unary, scalar, and comparison operations.
//! All shaders follow the same algorithm as CPU/CUDA implementations.

/// Element-wise shader module source (F32 only - WGSL doesn't support F64)
pub const ELEMENTWISE_SHADER: &str = r#"
// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// Binary Operations (element-wise: a[i] op b[i])
// ============================================================================

struct BinaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> binary_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> binary_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> binary_out: array<f32>;
@group(0) @binding(3) var<uniform> binary_params: BinaryParams;

@compute @workgroup_size(256)
fn add_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < binary_params.numel) {
        binary_out[gid] = binary_a[gid] + binary_b[gid];
    }
}

@compute @workgroup_size(256)
fn sub_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < binary_params.numel) {
        binary_out[gid] = binary_a[gid] - binary_b[gid];
    }
}

@compute @workgroup_size(256)
fn mul_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < binary_params.numel) {
        binary_out[gid] = binary_a[gid] * binary_b[gid];
    }
}

@compute @workgroup_size(256)
fn div_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < binary_params.numel) {
        binary_out[gid] = binary_a[gid] / binary_b[gid];
    }
}

@compute @workgroup_size(256)
fn pow_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < binary_params.numel) {
        binary_out[gid] = pow(binary_a[gid], binary_b[gid]);
    }
}

@compute @workgroup_size(256)
fn max_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < binary_params.numel) {
        binary_out[gid] = max(binary_a[gid], binary_b[gid]);
    }
}

@compute @workgroup_size(256)
fn min_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < binary_params.numel) {
        binary_out[gid] = min(binary_a[gid], binary_b[gid]);
    }
}

// ============================================================================
// Unary Operations (element-wise: out[i] = op(a[i]))
// ============================================================================

struct UnaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> unary_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> unary_out: array<f32>;
@group(0) @binding(2) var<uniform> unary_params: UnaryParams;

@compute @workgroup_size(256)
fn neg_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = -unary_a[gid];
    }
}

@compute @workgroup_size(256)
fn abs_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = abs(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn sqrt_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = sqrt(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn exp_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = exp(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn log_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = log(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn sin_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = sin(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn cos_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = cos(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn tan_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = tan(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn tanh_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = tanh(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn recip_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = 1.0 / unary_a[gid];
    }
}

@compute @workgroup_size(256)
fn square_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        let v = unary_a[gid];
        unary_out[gid] = v * v;
    }
}

@compute @workgroup_size(256)
fn floor_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = floor(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn ceil_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = ceil(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn round_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = round(unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn sign_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = sign(unary_a[gid]);
    }
}

// ============================================================================
// Scalar Operations (element-wise: out[i] = a[i] op scalar)
// ============================================================================

struct ScalarParams {
    numel: u32,
    scalar: f32,
}

@group(0) @binding(0) var<storage, read_write> scalar_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> scalar_out: array<f32>;
@group(0) @binding(2) var<uniform> scalar_params: ScalarParams;

@compute @workgroup_size(256)
fn add_scalar_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < scalar_params.numel) {
        scalar_out[gid] = scalar_a[gid] + scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn sub_scalar_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < scalar_params.numel) {
        scalar_out[gid] = scalar_a[gid] - scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn mul_scalar_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < scalar_params.numel) {
        scalar_out[gid] = scalar_a[gid] * scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn div_scalar_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < scalar_params.numel) {
        scalar_out[gid] = scalar_a[gid] / scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn pow_scalar_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < scalar_params.numel) {
        scalar_out[gid] = pow(scalar_a[gid], scalar_params.scalar);
    }
}

// ============================================================================
// Comparison Operations (element-wise: out[i] = (a[i] op b[i]) ? 1.0 : 0.0)
// ============================================================================

// Note: Using f32 output to match the current numr behavior

@group(0) @binding(0) var<storage, read_write> cmp_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> cmp_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> cmp_out: array<f32>;
@group(0) @binding(3) var<uniform> cmp_params: BinaryParams;

@compute @workgroup_size(256)
fn eq_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < cmp_params.numel) {
        cmp_out[gid] = select(0.0, 1.0, cmp_a[gid] == cmp_b[gid]);
    }
}

@compute @workgroup_size(256)
fn ne_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < cmp_params.numel) {
        cmp_out[gid] = select(0.0, 1.0, cmp_a[gid] != cmp_b[gid]);
    }
}

@compute @workgroup_size(256)
fn lt_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < cmp_params.numel) {
        cmp_out[gid] = select(0.0, 1.0, cmp_a[gid] < cmp_b[gid]);
    }
}

@compute @workgroup_size(256)
fn le_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < cmp_params.numel) {
        cmp_out[gid] = select(0.0, 1.0, cmp_a[gid] <= cmp_b[gid]);
    }
}

@compute @workgroup_size(256)
fn gt_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < cmp_params.numel) {
        cmp_out[gid] = select(0.0, 1.0, cmp_a[gid] > cmp_b[gid]);
    }
}

@compute @workgroup_size(256)
fn ge_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < cmp_params.numel) {
        cmp_out[gid] = select(0.0, 1.0, cmp_a[gid] >= cmp_b[gid]);
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

@compute @workgroup_size(256)
fn relu_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = max(0.0, unary_a[gid]);
    }
}

@compute @workgroup_size(256)
fn sigmoid_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        unary_out[gid] = 1.0 / (1.0 + exp(-unary_a[gid]));
    }
}

@compute @workgroup_size(256)
fn silu_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        let x = unary_a[gid];
        unary_out[gid] = x / (1.0 + exp(-x));
    }
}

@compute @workgroup_size(256)
fn gelu_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        let x = unary_a[gid];
        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_pi = 0.7978845608;
        let inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        unary_out[gid] = 0.5 * x * (1.0 + tanh(inner));
    }
}

// ============================================================================
// Clamp Operation
// ============================================================================

struct ClampParams {
    numel: u32,
    min_val: f32,
    max_val: f32,
}

@group(0) @binding(0) var<storage, read_write> clamp_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> clamp_out: array<f32>;
@group(0) @binding(2) var<uniform> clamp_params: ClampParams;

@compute @workgroup_size(256)
fn clamp_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < clamp_params.numel) {
        clamp_out[gid] = clamp(clamp_a[gid], clamp_params.min_val, clamp_params.max_val);
    }
}

// ============================================================================
// isnan / isinf Operations
// ============================================================================

@compute @workgroup_size(256)
fn isnan_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        let x = unary_a[gid];
        // WGSL: isnan not available, but x != x is true for NaN
        unary_out[gid] = select(0.0, 1.0, x != x);
    }
}

@compute @workgroup_size(256)
fn isinf_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < unary_params.numel) {
        let x = unary_a[gid];
        // Check if x is infinity: not NaN, but adding 1 doesn't change it (and not zero)
        let is_inf = (x == x) && (x + 1.0 == x) && (x != 0.0);
        unary_out[gid] = select(0.0, 1.0, is_inf);
    }
}

// ============================================================================
// Where Conditional (ternary: out[i] = cond[i] ? x[i] : y[i])
// ============================================================================

struct WhereParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> where_cond: array<f32>;
@group(0) @binding(1) var<storage, read_write> where_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> where_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> where_out: array<f32>;
@group(0) @binding(4) var<uniform> where_params: WhereParams;

@compute @workgroup_size(256)
fn where_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid < where_params.numel) {
        // Condition is true if non-zero
        where_out[gid] = select(where_y[gid], where_x[gid], where_cond[gid] != 0.0);
    }
}
"#;
