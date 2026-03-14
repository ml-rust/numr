//! Forward kernels for GEMM epilogue operations.
//!
//! matmul_bias_activation: C = activation(A @ B + bias)
//! matmul_bias_residual: C = A @ B + bias + residual

use crate::dtype::Element;
use crate::ops::GemmActivation;

/// Fused matmul + bias + activation kernel.
///
/// Computes `activation(A @ B + bias)` in a single pass:
/// 1. Initialize output with bias
/// 2. Accumulate matmul result (ikj order)
/// 3. Apply activation in-place
///
/// # Safety
/// - All pointers must be valid for the specified dimensions
/// - `out` must not alias with `a`, `b`, or `bias`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_activation_kernel<T: Element>(
    a: *const T,
    b: *const T,
    bias: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    activation: GemmActivation,
) {
    // For GemmActivation::None, just do matmul_bias (avoid activation dispatch overhead)
    if activation == GemmActivation::None {
        crate::runtime::cpu::kernels::matmul_bias_kernel(a, b, bias, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // SIMD dispatch for f32/f64 on x86_64: matmul_bias first, then apply activation via SIMD
    #[cfg(target_arch = "x86_64")]
    {
        use crate::dtype::DType;
        match T::DTYPE {
            DType::F32 => {
                matmul_bias_activation_simd_f32(
                    a as *const f32,
                    b as *const f32,
                    bias as *const f32,
                    out as *mut f32,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    activation,
                );
                return;
            }
            DType::F64 => {
                matmul_bias_activation_simd_f64(
                    a as *const f64,
                    b as *const f64,
                    bias as *const f64,
                    out as *mut f64,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    activation,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    matmul_bias_activation_scalar(a, b, bias, out, m, n, k, lda, ldb, ldc, activation);
}

/// Fused matmul + bias + residual kernel.
///
/// Computes `A @ B + bias + residual` in a single pass.
///
/// # Safety
/// - All pointers must be valid for the specified dimensions
/// - `out` must not alias with `a`, `b`, `bias`, or `residual`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_residual_kernel<T: Element>(
    a: *const T,
    b: *const T,
    bias: *const T,
    residual: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Initialize output with bias + residual
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = *bias.add(j) + *residual.add(i * ldc + j);
        }
    }

    // Accumulate matmul result (ikj order for cache locality)
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                let out_ptr = out.add(i * ldc + j);
                *out_ptr = *out_ptr + a_val * *b.add(kk * ldb + j);
            }
        }
    }
}

// ============================================================================
// SIMD-accelerated paths (matmul_bias then SIMD activation)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments, dead_code)]
unsafe fn matmul_bias_activation_simd_f32(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    activation: GemmActivation,
) {
    use super::super::simd::matmul;

    // Step 1: Compute matmul_bias into output buffer
    matmul::matmul_bias_f32(a, b, bias, out, m, n, k, lda, ldb, ldc);

    // Step 2: Apply activation in-place using SIMD
    let total = m * n;
    apply_activation_inplace_f32(out, total, activation);
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments, dead_code)]
unsafe fn matmul_bias_activation_simd_f64(
    a: *const f64,
    b: *const f64,
    bias: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    activation: GemmActivation,
) {
    use super::super::simd::matmul;

    // Step 1: Compute matmul_bias into output buffer
    matmul::matmul_bias_f64(a, b, bias, out, m, n, k, lda, ldb, ldc);

    // Step 2: Apply activation in-place using SIMD
    let total = m * n;
    apply_activation_inplace_f64(out, total, activation);
}

/// Apply activation in-place on f32 buffer using SIMD helpers.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn apply_activation_inplace_f32(buf: *mut f32, len: usize, activation: GemmActivation) {
    use super::super::simd::activations;

    match activation {
        GemmActivation::None => {}
        GemmActivation::ReLU => {
            // ReLU is simple: max(0, x) — use scalar for in-place
            for i in 0..len {
                let val = *buf.add(i);
                if val < 0.0 {
                    *buf.add(i) = 0.0;
                }
            }
        }
        GemmActivation::GELU => {
            // Use SIMD gelu (reads from buf, writes to buf — safe since non-overlapping access)
            activations::gelu_f32(buf as *const f32, buf, len);
        }
        GemmActivation::SiLU => {
            activations::silu_f32(buf as *const f32, buf, len);
        }
        GemmActivation::Sigmoid => {
            activations::sigmoid_f32(buf as *const f32, buf, len);
        }
        GemmActivation::Tanh => {
            for i in 0..len {
                *buf.add(i) = (*buf.add(i)).tanh();
            }
        }
    }
}

/// Apply activation in-place on f64 buffer using SIMD helpers.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn apply_activation_inplace_f64(buf: *mut f64, len: usize, activation: GemmActivation) {
    use super::super::simd::activations;

    match activation {
        GemmActivation::None => {}
        GemmActivation::ReLU => {
            for i in 0..len {
                let val = *buf.add(i);
                if val < 0.0 {
                    *buf.add(i) = 0.0;
                }
            }
        }
        GemmActivation::GELU => {
            activations::gelu_f64(buf as *const f64, buf, len);
        }
        GemmActivation::SiLU => {
            activations::silu_f64(buf as *const f64, buf, len);
        }
        GemmActivation::Sigmoid => {
            activations::sigmoid_f64(buf as *const f64, buf, len);
        }
        GemmActivation::Tanh => {
            for i in 0..len {
                *buf.add(i) = (*buf.add(i)).tanh();
            }
        }
    }
}

// ============================================================================
// Scalar fallback
// ============================================================================

#[allow(clippy::too_many_arguments, dead_code)]
unsafe fn matmul_bias_activation_scalar<T: Element>(
    a: *const T,
    b: *const T,
    bias: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    activation: GemmActivation,
) {
    // Initialize output with bias
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = *bias.add(j);
        }
    }

    // Accumulate matmul result (ikj order)
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                let out_ptr = out.add(i * ldc + j);
                *out_ptr = *out_ptr + a_val * *b.add(kk * ldb + j);
            }
        }
    }

    // Apply activation in-place
    apply_activation_scalar(out, m * n, activation);
}

/// Apply activation element-wise using scalar math (generic over Element).
#[allow(dead_code)]
unsafe fn apply_activation_scalar<T: Element>(buf: *mut T, len: usize, activation: GemmActivation) {
    match activation {
        GemmActivation::None => {}
        GemmActivation::ReLU => {
            for i in 0..len {
                let val = *buf.add(i);
                if val < T::zero() {
                    *buf.add(i) = T::zero();
                }
            }
        }
        GemmActivation::GELU => {
            // GELU needs float math — convert through f64
            for i in 0..len {
                let x = (*buf.add(i)).to_f64();
                let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
                let result = 0.5 * x * (1.0 + inner.tanh());
                *buf.add(i) = T::from_f64(result);
            }
        }
        GemmActivation::SiLU => {
            for i in 0..len {
                let x = (*buf.add(i)).to_f64();
                let result = x / (1.0 + (-x).exp());
                *buf.add(i) = T::from_f64(result);
            }
        }
        GemmActivation::Sigmoid => {
            for i in 0..len {
                let x = (*buf.add(i)).to_f64();
                let result = 1.0 / (1.0 + (-x).exp());
                *buf.add(i) = T::from_f64(result);
            }
        }
        GemmActivation::Tanh => {
            for i in 0..len {
                let x = (*buf.add(i)).to_f64();
                *buf.add(i) = T::from_f64(x.tanh());
            }
        }
    }
}
