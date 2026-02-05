//! CPU implementation of iterative solvers.
//!
//! This module provides CPU implementations of iterative solvers
//! using the generic algorithms in `algorithm::iterative_impl`.

use super::{CpuClient, CpuRuntime};
use crate::algorithm::iterative::{
    BiCgStabOptions, BiCgStabResult, GmresOptions, GmresResult, IterativeSolvers,
};
use crate::algorithm::iterative_impl::{bicgstab_impl, gmres_impl};
use crate::error::Result;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

impl IterativeSolvers<CpuRuntime> for CpuClient {
    fn gmres(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: GmresOptions,
    ) -> Result<GmresResult<CpuRuntime>> {
        gmres_impl(self, a, b, x0, options)
    }

    fn bicgstab(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: BiCgStabOptions,
    ) -> Result<BiCgStabResult<CpuRuntime>> {
        bicgstab_impl(self, a, b, x0, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::iterative::PreconditionerType;
    use crate::runtime::Runtime;

    fn get_client() -> CpuClient {
        let device = CpuRuntime::default_device();
        CpuRuntime::default_client(&device)
    }

    /// Create 1D Laplacian (tridiagonal) matrix:
    /// [ 2, -1,  0,  0, ...]
    /// [-1,  2, -1,  0, ...]
    /// [ 0, -1,  2, -1, ...]
    /// ...
    fn create_1d_laplacian(
        n: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> CsrData<CpuRuntime> {
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptrs.push(0i64);
        for i in 0..n {
            // Entry (i, i-1) = -1 (if i > 0)
            if i > 0 {
                col_indices.push((i - 1) as i64);
                values.push(-1.0f64);
            }
            // Entry (i, i) = 2
            col_indices.push(i as i64);
            values.push(2.0f64);
            // Entry (i, i+1) = -1 (if i < n-1)
            if i < n - 1 {
                col_indices.push((i + 1) as i64);
                values.push(-1.0f64);
            }
            row_ptrs.push(col_indices.len() as i64);
        }

        let row_ptrs_tensor =
            Tensor::<CpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], device);
        let col_indices_tensor =
            Tensor::<CpuRuntime>::from_slice(&col_indices, &[col_indices.len()], device);
        let values_tensor = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], device);

        CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
            .expect("CSR creation should succeed")
    }

    #[test]
    fn test_gmres_tridiagonal() {
        let client = get_client();
        let device = &client.device;

        // Create small 5x5 Laplacian
        let n = 5;
        let a = create_1d_laplacian(n, device);

        // b = [1, 0, 0, 0, 1] - symmetric RHS
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 1.0], &[n], device);

        let options = GmresOptions {
            max_iter: 100,
            restart: 10,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        };

        let result = client
            .gmres(&a, &b, None, options)
            .expect("GMRES should succeed");

        assert!(result.converged, "GMRES should converge");
        assert!(
            result.iterations <= 10,
            "GMRES should converge quickly on tridiagonal"
        );
        assert!(
            result.residual_norm < 1e-8,
            "Residual should be small: {}",
            result.residual_norm
        );

        // Verify solution: compute ||b - Ax||
        let ax = a.spmv(&result.solution).expect("spmv should work");
        use crate::ops::BinaryOps;
        let residual = client.sub(&b, &ax).expect("sub should work");
        let res_data: Vec<f64> = residual.to_vec();
        let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
    }

    #[test]
    fn test_gmres_with_ilu0() {
        let client = get_client();
        let device = &client.device;

        // Larger system to show benefit of preconditioning
        let n = 20;
        let a = create_1d_laplacian(n, device);

        // Random-ish RHS
        let b_data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

        // Use default restart (30) for non-preconditioned to allow convergence
        let options_no_precond = GmresOptions {
            max_iter: 100,
            restart: 30, // Larger restart needed for non-preconditioned
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        };

        // ILU(0) on tridiagonal is exact LU, so even small restart works
        let options_ilu = GmresOptions {
            max_iter: 100,
            restart: 10, // Small restart - ILU makes this sufficient
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::Ilu0,
        };

        let result_no_precond = client
            .gmres(&a, &b, None, options_no_precond)
            .expect("GMRES should succeed");
        let result_ilu = client
            .gmres(&a, &b, None, options_ilu)
            .expect("GMRES with ILU should succeed");

        assert!(result_no_precond.converged, "GMRES should converge");
        assert!(result_ilu.converged, "GMRES+ILU should converge");

        // ILU(0) should converge faster (or at least same speed) for well-structured matrices
        // For tridiagonal, ILU(0) is exact LU, so convergence is immediate
        assert!(
            result_ilu.iterations <= result_no_precond.iterations,
            "ILU should help convergence: {} vs {}",
            result_ilu.iterations,
            result_no_precond.iterations
        );
    }

    #[test]
    fn test_gmres_identity_matrix() {
        let client = get_client();
        let device = &client.device;

        // Identity matrix - should converge in 1 iteration
        let n = 5;
        let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4, 5], &[n + 1], device);
        let col_indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4], &[n], device);
        let values = Tensor::<CpuRuntime>::from_slice(&[1.0f64; 5], &[n], device);
        let a = CsrData::new(row_ptrs, col_indices, values, [n, n])
            .expect("CSR creation should succeed");

        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[n], device);

        let options = GmresOptions::default();
        let result = client
            .gmres(&a, &b, None, options)
            .expect("GMRES should succeed");

        assert!(result.converged);
        assert_eq!(
            result.iterations, 1,
            "Identity should converge in 1 iteration"
        );

        // Solution should equal b
        let x_data: Vec<f64> = result.solution.to_vec();
        let b_data: Vec<f64> = b.to_vec();
        for i in 0..n {
            assert!(
                (x_data[i] - b_data[i]).abs() < 1e-10,
                "x[{}] = {} != b[{}] = {}",
                i,
                x_data[i],
                i,
                b_data[i]
            );
        }
    }

    #[test]
    fn test_bicgstab_tridiagonal() {
        let client = get_client();
        let device = &client.device;

        let n = 5;
        let a = create_1d_laplacian(n, device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 1.0], &[n], device);

        let options = BiCgStabOptions {
            max_iter: 100,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        };

        let result = client
            .bicgstab(&a, &b, None, options)
            .expect("BiCGSTAB should succeed");

        assert!(result.converged, "BiCGSTAB should converge");
        assert!(
            result.residual_norm < 1e-8,
            "Residual: {}",
            result.residual_norm
        );

        // Verify solution
        let ax = a.spmv(&result.solution).expect("spmv should work");
        use crate::ops::BinaryOps;
        let residual = client.sub(&b, &ax).expect("sub should work");
        let res_data: Vec<f64> = residual.to_vec();
        let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
    }

    #[test]
    fn test_gmres_zero_rhs() {
        let client = get_client();
        let device = &client.device;

        let n = 5;
        let a = create_1d_laplacian(n, device);
        let b = Tensor::<CpuRuntime>::zeros(&[n], crate::dtype::DType::F64, device);

        let options = GmresOptions::default();
        let result = client
            .gmres(&a, &b, None, options)
            .expect("GMRES should succeed");

        // Zero RHS should give zero solution immediately
        assert!(result.converged);
        assert_eq!(result.iterations, 0);

        let x_data: Vec<f64> = result.solution.to_vec();
        for (i, &xi) in x_data.iter().enumerate() {
            assert!(xi.abs() < 1e-14, "x[{}] = {} should be zero", i, xi);
        }
    }

    #[test]
    fn test_gmres_with_initial_guess() {
        let client = get_client();
        let device = &client.device;

        let n = 5;
        let a = create_1d_laplacian(n, device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 1.0], &[n], device);

        // First solve without initial guess
        let result1 = client
            .gmres(&a, &b, None, GmresOptions::default())
            .expect("GMRES should succeed");
        assert!(result1.converged);

        // Use the solution as initial guess - should converge immediately
        let result2 = client
            .gmres(&a, &b, Some(&result1.solution), GmresOptions::default())
            .expect("GMRES should succeed");
        assert!(result2.converged);
        assert!(
            result2.iterations <= 1,
            "With good initial guess, should converge in 0-1 iterations"
        );
    }
}
