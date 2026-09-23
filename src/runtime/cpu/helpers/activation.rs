//! Activation operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::runtime::{ensure_contiguous, row_layout};
use crate::tensor::Tensor;

/// Activation operation kind for kernel dispatch
#[derive(Copy, Clone)]
pub enum ActivationOp {
    Relu,
    Sigmoid,
    Silu,
    Gelu,
}

/// Parametric activation operation kind (activations that take a scalar parameter)
#[derive(Copy, Clone)]
pub enum ParametricActivationOp {
    /// LeakyReLU: x if x > 0, else negative_slope * x
    LeakyRelu,
    /// ELU: x if x > 0, else alpha * (exp(x) - 1)
    Elu,
}

/// Helper for activation operations (relu, sigmoid, silu, gelu)
pub fn activation_op_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    op: ActivationOp,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let total = a.numel();
    let (rows, row_len) = row_layout(a.shape(), total);
    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        let a_ptr = a_ptr as *const T;
        let out_ptr = out_ptr as *mut T;
        unsafe {
            for row in 0..rows {
                let off = row * row_len;
                match op {
                    ActivationOp::Relu => kernels::relu_kernel::<T>(
                        a_ptr.add(off),
                        out_ptr.add(off),
                        row_len,
                    ),
                    ActivationOp::Sigmoid => kernels::sigmoid_kernel::<T>(
                        a_ptr.add(off),
                        out_ptr.add(off),
                        row_len,
                    ),
                    ActivationOp::Silu => kernels::silu_kernel::<T>(
                        a_ptr.add(off),
                        out_ptr.add(off),
                        row_len,
                    ),
                    ActivationOp::Gelu => kernels::gelu_kernel::<T>(
                        a_ptr.add(off),
                        out_ptr.add(off),
                        row_len,
                    ),
                }
            }
        }
    }, op_name);

    Ok(out)
}

/// Fused activation-mul operation kind
#[derive(Copy, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum FusedActivationMulOp {
    SiluMul,
    GeluMul,
    ReluMul,
    SigmoidMul,
}

/// Helper for fused activation-mul operations: activation(a) * b
pub fn fused_activation_mul_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op: FusedActivationMulOp,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    if b.dtype() != dtype {
        return Err(crate::error::Error::DTypeMismatch {
            lhs: dtype,
            rhs: b.dtype(),
        });
    }
    if a.shape() != b.shape() {
        return Err(crate::error::Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let total = a.numel();
    let (rows, row_len) = row_layout(a.shape(), total);
    let a_ptr = a_contig.ptr();
    let b_ptr = b_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        let a_ptr = a_ptr as *const T;
        let b_ptr = b_ptr as *const T;
        let out_ptr = out_ptr as *mut T;
        unsafe {
            for row in 0..rows {
                let off = row * row_len;
                match op {
                    FusedActivationMulOp::SiluMul => kernels::silu_mul_kernel::<T>(
                        a_ptr.add(off), b_ptr.add(off), out_ptr.add(off), row_len,
                    ),
                    FusedActivationMulOp::GeluMul => kernels::gelu_mul_kernel::<T>(
                        a_ptr.add(off), b_ptr.add(off), out_ptr.add(off), row_len,
                    ),
                    FusedActivationMulOp::ReluMul => kernels::relu_mul_kernel::<T>(
                        a_ptr.add(off), b_ptr.add(off), out_ptr.add(off), row_len,
                    ),
                    FusedActivationMulOp::SigmoidMul => kernels::sigmoid_mul_kernel::<T>(
                        a_ptr.add(off), b_ptr.add(off), out_ptr.add(off), row_len,
                    ),
                }
            }
        }
    }, op_name);

    Ok(out)
}

/// Helper for parametric activation operations (leaky_relu, elu)
///
/// These activations take a single f64 parameter in addition to the input tensor.
pub fn parametric_activation_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    op: ParametricActivationOp,
    param: f64,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let total = a.numel();
    let (rows, row_len) = row_layout(a.shape(), total);
    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        let a_ptr = a_ptr as *const T;
        let out_ptr = out_ptr as *mut T;
        unsafe {
            for row in 0..rows {
                let off = row * row_len;
                match op {
                    ParametricActivationOp::LeakyRelu => kernels::leaky_relu_kernel::<T>(
                        a_ptr.add(off),
                        out_ptr.add(off),
                        row_len,
                        param,
                    ),
                    ParametricActivationOp::Elu => kernels::elu_kernel::<T>(
                        a_ptr.add(off),
                        out_ptr.add(off),
                        row_len,
                        param,
                    ),
                }
            }
        }
    }, op_name);

    Ok(out)
}

/// Helper for leaky_relu activation
#[inline]
pub fn leaky_relu_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    negative_slope: f64,
) -> Result<Tensor<CpuRuntime>> {
    parametric_activation_impl(
        client,
        a,
        ParametricActivationOp::LeakyRelu,
        negative_slope,
        "leaky_relu",
    )
}

/// Helper for ELU activation
#[inline]
pub fn elu_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    alpha: f64,
) -> Result<Tensor<CpuRuntime>> {
    parametric_activation_impl(client, a, ParametricActivationOp::Elu, alpha, "elu")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::ActivationOps;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    /// Positive and negative values, not aligned to any particular SIMD
    /// lane width or period.
    fn row_data(w: usize, seed: i32) -> Vec<f32> {
        (0..w)
            .map(|i| ((i as i32 + seed) % 13 - 6) as f32 * 0.37)
            .collect()
    }

    /// Row `r` of a `[B, 1, W]` batched call must be bitwise identical to
    /// that row computed alone as `[1, 1, W]`. `W` in {8, 16, 17} straddles
    /// `SIMD_THRESHOLD` (32) once batched by `B` in {1, 2, 3, 5}, and 17 is
    /// not a multiple of any SIMD lane width, so the scalar tail is
    /// exercised too.
    fn assert_row_invariant<F>(op: F)
    where
        F: Fn(&CpuClient, &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        for &w in &[8usize, 16, 17] {
            for &b in &[1usize, 2, 3, 5] {
                let mut data = Vec::with_capacity(b * w);
                for r in 0..b {
                    data.extend(row_data(w, r as i32));
                }
                let batched = Tensor::<CpuRuntime>::from_slice(&data, &[b, 1, w], &device).unwrap();
                let batched_out: Vec<f32> = op(&client, &batched).unwrap().to_vec();

                for r in 0..b {
                    let row = &data[r * w..(r + 1) * w];
                    let solo = Tensor::<CpuRuntime>::from_slice(row, &[1, 1, w], &device).unwrap();
                    let solo_out: Vec<f32> = op(&client, &solo).unwrap().to_vec();
                    let batched_row = &batched_out[r * w..(r + 1) * w];

                    for i in 0..w {
                        assert_eq!(
                            batched_row[i].to_bits(),
                            solo_out[i].to_bits(),
                            "w={w} b={b} row={r} idx={i}: batched={} solo={}",
                            batched_row[i],
                            solo_out[i],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_silu_row_invariant() {
        assert_row_invariant(|client, x| client.silu(x));
    }

    #[test]
    fn test_sigmoid_row_invariant() {
        assert_row_invariant(|client, x| client.sigmoid(x));
    }

    /// Same invariant for `elu`, whose negative branch uses `exp`: SIMD and
    /// scalar `exp` are not bit-identical, so a row's result bits must not
    /// depend on how many other rows share the call.
    #[test]
    fn test_elu_row_invariant() {
        assert_row_invariant(|client, x| client.elu(x, 1.0));
    }

    /// Same invariant for the fused `silu(a) * b` path, which reads two
    /// inputs per row instead of one.
    #[test]
    fn test_silu_mul_row_invariant() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        for &w in &[8usize, 16, 17] {
            for &b in &[1usize, 2, 3, 5] {
                let mut a_data = Vec::with_capacity(b * w);
                let mut b_data = Vec::with_capacity(b * w);
                for r in 0..b {
                    a_data.extend(row_data(w, r as i32));
                    b_data.extend(row_data(w, r as i32 + 100));
                }
                let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[b, 1, w], &device).unwrap();
                let bt = Tensor::<CpuRuntime>::from_slice(&b_data, &[b, 1, w], &device).unwrap();
                let batched_out: Vec<f32> = client.silu_mul(&a, &bt).unwrap().to_vec();

                for r in 0..b {
                    let a_row = &a_data[r * w..(r + 1) * w];
                    let b_row = &b_data[r * w..(r + 1) * w];
                    let a_solo =
                        Tensor::<CpuRuntime>::from_slice(a_row, &[1, 1, w], &device).unwrap();
                    let b_solo =
                        Tensor::<CpuRuntime>::from_slice(b_row, &[1, 1, w], &device).unwrap();
                    let solo_out: Vec<f32> = client.silu_mul(&a_solo, &b_solo).unwrap().to_vec();
                    let batched_row = &batched_out[r * w..(r + 1) * w];

                    for i in 0..w {
                        assert_eq!(
                            batched_row[i].to_bits(),
                            solo_out[i].to_bits(),
                            "w={w} b={b} row={r} idx={i}: batched={} solo={}",
                            batched_row[i],
                            solo_out[i],
                        );
                    }
                }
            }
        }
    }
}
