//! Autograd-aware `matmul_wide`: the product in the accumulator dtype.

use super::ops::MatmulWideBackward;
use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};

/// Matrix multiplication written in the accumulator dtype: `z = a @ b` with
/// F16/BF16 operands producing an F32 `z` (I8 producing I32, every other
/// dtype its own). See [`crate::ops::MatmulOps::matmul_wide`].
///
/// The backward casts the incoming gradient to the operand dtype and applies
/// the `matmul` formulas, so `a` and `b` receive gradients in their own dtype.
pub fn var_matmul_wide<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.matmul_wide(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = MatmulWideBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            b.tensor().clone(),
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(all(test, feature = "f16"))]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::autograd::var_ops::{var_mul, var_sum};
    use crate::ops::{MatmulOps, TypeConversionOps};
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    const M: usize = 5;
    const K: usize = 7;
    const N: usize = 6;

    /// Values on a coarse grid, so F16 and BF16 hold them exactly and every
    /// backend multiplies the same numbers.
    fn grid(len: usize, seed: usize) -> Vec<f64> {
        (0..len)
            .map(|i| (((i * 7 + seed * 3) % 17) as f64) * 0.125 - 1.0)
            .collect()
    }

    fn cpu() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    fn cpu_tensor(
        client: &CpuClient,
        device: &CpuDevice,
        data: &[f64],
        shape: &[usize],
        dtype: DType,
    ) -> Tensor<CpuRuntime> {
        let t = Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap();
        client.cast(&t, dtype).unwrap()
    }

    fn assert_close(got: &[f32], want: &[f64], rel: f64, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = rel * w.abs().max(1.0);
            assert!(
                (f64::from(*g) - w).abs() <= tol,
                "{label}: element {i}: {g} vs {w}"
            );
        }
    }

    /// Forward: the F32 output equals `cast(matmul(cast(a, F32), cast(b, F32)))`
    /// — here the outer cast is the identity, so the two F32 tensors are
    /// compared directly.
    #[test]
    fn test_var_matmul_wide_forward_matches_f32_route() {
        let (client, device) = cpu();
        for dtype in [DType::F16, DType::BF16] {
            let a = cpu_tensor(&client, &device, &grid(M * K, 1), &[M, K], dtype);
            let b = cpu_tensor(&client, &device, &grid(K * N, 2), &[K, N], dtype);

            let out = var_matmul_wide(
                &Var::new(a.clone(), false),
                &Var::new(b.clone(), false),
                &client,
            )
            .unwrap();
            assert_eq!(out.tensor().dtype(), DType::F32);
            assert!(!out.requires_grad());

            let a32 = client.cast(&a, DType::F32).unwrap();
            let b32 = client.cast(&b, DType::F32).unwrap();
            let want = client.matmul(&a32, &b32).unwrap();
            let want = client.cast(&want, DType::F32).unwrap();
            let want: Vec<f64> = want.to_vec::<f32>().iter().map(|&v| f64::from(v)).collect();
            assert_close(&out.tensor().to_vec::<f32>(), &want, 1e-5, "forward");
        }
    }

    /// Loss `L = sum(w * matmul_wide(a, b))` with a fixed F32 weight `w`.
    /// `dL/da = w @ b^T` and `dL/db = a^T @ w`, and finite differences in F64
    /// on the same loss recover them without using the formulas. The F64
    /// route casts to F32 first, the same route the forward takes, and the
    /// grid values make every cast exact.
    #[test]
    fn test_var_matmul_wide_backward_matches_finite_differences() {
        let (client, device) = cpu();
        let a_data = grid(M * K, 1);
        let b_data = grid(K * N, 2);
        let w_data = grid(M * N, 3);

        // Loss in F64 on CPU for arbitrary (a, b), through the F32 route.
        let loss = |a: &[f64], b: &[f64]| -> f64 {
            let a64 = Tensor::<CpuRuntime>::from_slice(a, &[M, K], &device).unwrap();
            let b64 = Tensor::<CpuRuntime>::from_slice(b, &[K, N], &device).unwrap();
            let a32 = client.cast(&a64, DType::F32).unwrap();
            let b32 = client.cast(&b64, DType::F32).unwrap();
            let prod = client.matmul(&a32, &b32).unwrap();
            let prod: Vec<f32> = client
                .cast(&prod, DType::F64)
                .unwrap()
                .to_vec::<f64>()
                .iter()
                .map(|&v| v as f32)
                .collect();
            prod.iter()
                .zip(&w_data)
                .map(|(p, w)| f64::from(*p) * w)
                .sum()
        };

        // Central differences with a step the F16 grid can absorb exactly
        // in the perturbed operand: 2^-3 keeps every value on the grid.
        let h = 0.125f64;
        let fd = |data: &[f64], other: &[f64], is_a: bool| -> Vec<f64> {
            (0..data.len())
                .map(|i| {
                    let mut plus = data.to_vec();
                    let mut minus = data.to_vec();
                    plus[i] += h;
                    minus[i] -= h;
                    let (lp, lm) = if is_a {
                        (loss(&plus, other), loss(&minus, other))
                    } else {
                        (loss(other, &plus), loss(other, &minus))
                    };
                    (lp - lm) / (2.0 * h)
                })
                .collect()
        };
        let fd_a = fd(&a_data, &b_data, true);
        let fd_b = fd(&b_data, &a_data, false);

        for dtype in [DType::F16, DType::BF16] {
            let a = Var::new(cpu_tensor(&client, &device, &a_data, &[M, K], dtype), true);
            let b = Var::new(cpu_tensor(&client, &device, &b_data, &[K, N], dtype), true);
            let w = Var::new(
                cpu_tensor(&client, &device, &w_data, &[M, N], DType::F32),
                false,
            );

            let prod = var_matmul_wide(&a, &b, &client).unwrap();
            let weighted = var_mul(&prod, &w, &client).unwrap();
            let l = var_sum(&weighted, &[0, 1], false, &client).unwrap();
            let grads = backward(&l, &client).unwrap();

            let grad_a = grads.get(a.id()).unwrap();
            assert_eq!(grad_a.dtype(), dtype, "grad dtype [{dtype:?}]");
            let grad_a: Vec<f32> = client.cast(grad_a, DType::F32).unwrap().to_vec();
            // The gradient itself is stored in the half dtype: one rounding
            // of a K-term (N-term) sum of exact products.
            let rel = if dtype == DType::F16 { 2e-3 } else { 1.6e-2 };
            assert_close(&grad_a, &fd_a, rel, &format!("dL/da [{dtype:?}]"));

            let grad_b = grads.get(b.id()).unwrap();
            assert_eq!(grad_b.dtype(), dtype, "grad dtype [{dtype:?}]");
            let grad_b: Vec<f32> = client.cast(grad_b, DType::F32).unwrap().to_vec();
            assert_close(&grad_b, &fd_b, rel, &format!("dL/db [{dtype:?}]"));
        }
    }

    /// A non-widening dtype: the output and both gradients stay F32, matching
    /// `var_matmul`.
    #[test]
    fn test_var_matmul_wide_f32_matches_var_matmul() {
        use crate::autograd::var_ops::var_matmul;

        let (client, device) = cpu();
        let a_t = cpu_tensor(&client, &device, &grid(M * K, 1), &[M, K], DType::F32);
        let b_t = cpu_tensor(&client, &device, &grid(K * N, 2), &[K, N], DType::F32);

        let (a, b) = (Var::new(a_t.clone(), true), Var::new(b_t.clone(), true));
        let wide = var_matmul_wide(&a, &b, &client).unwrap();
        let l = var_sum(&wide, &[0, 1], false, &client).unwrap();
        let wide_grads = backward(&l, &client).unwrap();

        let (a2, b2) = (Var::new(a_t, true), Var::new(b_t, true));
        let plain = var_matmul(&a2, &b2, &client).unwrap();
        let l2 = var_sum(&plain, &[0, 1], false, &client).unwrap();
        let plain_grads = backward(&l2, &client).unwrap();

        assert_eq!(wide.tensor().dtype(), DType::F32);
        assert_eq!(
            wide.tensor().to_vec::<f32>(),
            plain.tensor().to_vec::<f32>()
        );
        assert_eq!(
            wide_grads.get(a.id()).unwrap().to_vec::<f32>(),
            plain_grads.get(a2.id()).unwrap().to_vec::<f32>()
        );
        assert_eq!(
            wide_grads.get(b.id()).unwrap().to_vec::<f32>(),
            plain_grads.get(b2.id()).unwrap().to_vec::<f32>()
        );
    }

    /// CUDA forward and backward against CPU on the same F16 operands.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_var_matmul_wide_cuda_matches_cpu() {
        use crate::runtime::cuda::{CudaDevice, CudaRuntime, is_cuda_available};

        if !is_cuda_available() {
            return;
        }
        let (cpu_client, cpu_device) = cpu();
        let cuda_device = CudaDevice::new(0);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        let a_data = grid(M * K, 1);
        let b_data = grid(K * N, 2);

        for dtype in [DType::F16, DType::BF16] {
            let a = Var::new(
                cpu_tensor(&cpu_client, &cpu_device, &a_data, &[M, K], dtype),
                true,
            );
            let b = Var::new(
                cpu_tensor(&cpu_client, &cpu_device, &b_data, &[K, N], dtype),
                true,
            );
            let out = var_matmul_wide(&a, &b, &cpu_client).unwrap();
            let l = var_sum(&out, &[0, 1], false, &cpu_client).unwrap();
            let grads = backward(&l, &cpu_client).unwrap();
            let cpu_out = out.tensor().to_vec::<f32>();
            let cpu_ga: Vec<f32> = cpu_client
                .cast(grads.get(a.id()).unwrap(), DType::F32)
                .unwrap()
                .to_vec();
            let cpu_gb: Vec<f32> = cpu_client
                .cast(grads.get(b.id()).unwrap(), DType::F32)
                .unwrap()
                .to_vec();

            let a64 = Tensor::<CudaRuntime>::from_slice(&a_data, &[M, K], &cuda_device).unwrap();
            let b64 = Tensor::<CudaRuntime>::from_slice(&b_data, &[K, N], &cuda_device).unwrap();
            let a = Var::new(cuda_client.cast(&a64, dtype).unwrap(), true);
            let b = Var::new(cuda_client.cast(&b64, dtype).unwrap(), true);
            let out = var_matmul_wide(&a, &b, &cuda_client).unwrap();
            assert_eq!(out.tensor().dtype(), DType::F32);
            let l = var_sum(&out, &[0, 1], false, &cuda_client).unwrap();
            let grads = backward(&l, &cuda_client).unwrap();
            let ga = grads.get(a.id()).unwrap();
            let gb = grads.get(b.id()).unwrap();
            assert_eq!(ga.dtype(), dtype);
            assert_eq!(gb.dtype(), dtype);
            let cuda_ga: Vec<f32> = cuda_client.cast(ga, DType::F32).unwrap().to_vec();
            let cuda_gb: Vec<f32> = cuda_client.cast(gb, DType::F32).unwrap().to_vec();

            let as_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
            assert_close(
                &out.tensor().to_vec::<f32>(),
                &as_f64(&cpu_out),
                1e-5,
                "forward",
            );
            assert_close(&cuda_ga, &as_f64(&cpu_ga), 1e-5, "dL/da");
            assert_close(&cuda_gb, &as_f64(&cpu_gb), 1e-5, "dL/db");
        }
    }
}
