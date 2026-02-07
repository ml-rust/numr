//! Integration tests for iterative solvers and sparse eigensolvers

#![cfg(feature = "sparse")]

use numr::algorithm::iterative::{
    CgOptions, CgsOptions, IterativeSolvers, JacobiOptions, LgmresOptions, MinresOptions,
    PreconditionerType, QmrOptions, SorOptions,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::sparse::CsrData;
use numr::tensor::Tensor;

fn get_client() -> CpuClient {
    let device = CpuRuntime::default_device();
    CpuRuntime::default_client(&device)
}

/// Create 1D Laplacian (SPD tridiagonal): diag=2, off-diag=-1
fn create_1d_laplacian(n: usize, device: &<CpuRuntime as Runtime>::Device) -> CsrData<CpuRuntime> {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0i64);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as i64);
            values.push(-1.0f64);
        }
        col_indices.push(i as i64);
        values.push(2.0f64);
        if i < n - 1 {
            col_indices.push((i + 1) as i64);
            values.push(-1.0f64);
        }
        row_ptrs.push(col_indices.len() as i64);
    }

    let row_ptrs_tensor = Tensor::<CpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], device);
    let col_indices_tensor =
        Tensor::<CpuRuntime>::from_slice(&col_indices, &[col_indices.len()], device);
    let values_tensor = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], device);

    CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
        .expect("CSR creation should succeed")
}

/// Create a non-symmetric sparse matrix for CGS/Arnoldi testing
/// Convection-diffusion: tridiagonal with asymmetric off-diags
fn create_nonsymmetric(n: usize, device: &<CpuRuntime as Runtime>::Device) -> CsrData<CpuRuntime> {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0i64);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as i64);
            values.push(-1.0f64); // diffusion
        }
        col_indices.push(i as i64);
        values.push(3.0f64); // diagonal dominance
        if i < n - 1 {
            col_indices.push((i + 1) as i64);
            values.push(-1.5f64); // convection (asymmetric)
        }
        row_ptrs.push(col_indices.len() as i64);
    }

    let row_ptrs_tensor = Tensor::<CpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], device);
    let col_indices_tensor =
        Tensor::<CpuRuntime>::from_slice(&col_indices, &[col_indices.len()], device);
    let values_tensor = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], device);

    CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
        .expect("CSR creation should succeed")
}

// ============================================================================
// CG Tests
// ============================================================================

#[test]
fn test_cg_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .cg(&a, &b, None, CgOptions::default())
        .expect("CG should succeed");

    assert!(result.converged, "CG should converge on SPD Laplacian");
    assert!(
        result.residual_norm < 1e-8,
        "Residual too large: {}",
        result.residual_norm
    );

    // Verify solution
    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_cg_with_preconditioner() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 20;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result_plain = client
        .cg(&a, &b, None, CgOptions::default())
        .expect("CG should succeed");

    let result_ilu = client
        .cg(
            &a,
            &b,
            None,
            CgOptions {
                preconditioner: PreconditionerType::Ilu0,
                ..Default::default()
            },
        )
        .expect("CG+ILU should succeed");

    assert!(result_plain.converged);
    assert!(result_ilu.converged);
    assert!(
        result_ilu.iterations <= result_plain.iterations,
        "ILU should help: {} vs {}",
        result_ilu.iterations,
        result_plain.iterations
    );
}

#[test]
fn test_cg_identity() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 5;
    let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4, 5], &[n + 1], device);
    let col_indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4], &[n], device);
    let values = Tensor::<CpuRuntime>::from_slice(&[1.0f64; 5], &[n], device);
    let a = CsrData::new(row_ptrs, col_indices, values, [n, n]).unwrap();

    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[n], device);

    let result = client
        .cg(&a, &b, None, CgOptions::default())
        .expect("CG should succeed");

    assert!(result.converged);
    assert_eq!(result.iterations, 1, "Identity should converge in 1 iter");

    let x_data: Vec<f64> = result.solution.to_vec();
    for (i, &val) in x_data.iter().enumerate() {
        assert!((val - (i as f64 + 1.0)).abs() < 1e-10);
    }
}

// ============================================================================
// CGS Tests
// ============================================================================

#[test]
fn test_cgs_nonsymmetric() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_nonsymmetric(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .cgs(&a, &b, None, CgsOptions::default())
        .expect("CGS should succeed");

    assert!(result.converged, "CGS should converge");
    assert!(
        result.residual_norm < 1e-8,
        "Residual: {}",
        result.residual_norm
    );

    // Verify
    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_cgs_laplacian() {
    // CGS also works on SPD systems
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b = Tensor::<CpuRuntime>::from_slice(
        &[1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[n],
        device,
    );

    let result = client
        .cgs(&a, &b, None, CgsOptions::default())
        .expect("CGS should succeed");

    assert!(result.converged, "CGS should converge on Laplacian");
}

// ============================================================================
// MINRES Tests
// ============================================================================

#[test]
fn test_minres_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .minres(&a, &b, None, MinresOptions::default())
        .expect("MINRES should succeed");

    assert!(result.converged, "MINRES should converge on SPD Laplacian");
    assert!(
        result.residual_norm < 1e-8,
        "Residual: {}",
        result.residual_norm
    );

    // Verify
    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_minres_symmetric_indefinite() {
    // Create symmetric indefinite matrix: [-2, 1; 1, -2] shifted to have mixed eigenvalues
    // A = [[1, 2], [2, -1]] — symmetric, eigenvalues ≈ 2.236 and -2.236
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 2;
    let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 4], &[3], device);
    let col_indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 0, 1], &[4], device);
    let values = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 2.0, -1.0], &[4], device);
    let a = CsrData::new(row_ptrs, col_indices, values, [n, n]).unwrap();

    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 1.0], &[n], device);

    let result = client
        .minres(&a, &b, None, MinresOptions::default())
        .expect("MINRES should succeed on indefinite system");

    assert!(
        result.converged,
        "MINRES should converge on symmetric indefinite"
    );

    // Verify
    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-6, "Verification residual: {}", res_norm);
}

// ============================================================================
// Solver comparison test
// ============================================================================

#[test]
fn test_cg_vs_gmres_on_spd() {
    // CG and GMRES should produce the same solution on SPD systems
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let cg_result = client
        .cg(&a, &b, None, CgOptions::default())
        .expect("CG should succeed");

    let gmres_result = client
        .gmres(
            &a,
            &b,
            None,
            numr::algorithm::iterative::GmresOptions::default(),
        )
        .expect("GMRES should succeed");

    assert!(cg_result.converged);
    assert!(gmres_result.converged);

    let cg_x: Vec<f64> = cg_result.solution.to_vec();
    let gmres_x: Vec<f64> = gmres_result.solution.to_vec();
    for i in 0..n {
        assert!(
            (cg_x[i] - gmres_x[i]).abs() < 1e-6,
            "CG and GMRES solutions differ at [{}]: {} vs {}",
            i,
            cg_x[i],
            gmres_x[i]
        );
    }
}

// ============================================================================
// Jacobi Tests
// ============================================================================

/// Create diagonally dominant matrix: 4I + Laplacian
fn create_diag_dominant(n: usize, device: &<CpuRuntime as Runtime>::Device) -> CsrData<CpuRuntime> {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0i64);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as i64);
            values.push(-1.0f64);
        }
        col_indices.push(i as i64);
        values.push(6.0f64);
        if i < n - 1 {
            col_indices.push((i + 1) as i64);
            values.push(-1.0f64);
        }
        row_ptrs.push(col_indices.len() as i64);
    }

    let row_ptrs_tensor = Tensor::<CpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], device);
    let col_indices_tensor =
        Tensor::<CpuRuntime>::from_slice(&col_indices, &[col_indices.len()], device);
    let values_tensor = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], device);

    CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
        .expect("CSR creation should succeed")
}

#[test]
fn test_jacobi_diag_dominant() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_diag_dominant(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .jacobi(
            &a,
            &b,
            None,
            JacobiOptions {
                max_iter: 500,
                omega: 2.0 / 3.0,
                rtol: 1e-10,
                atol: 1e-14,
            },
        )
        .expect("Jacobi should succeed");

    assert!(result.converged, "Jacobi should converge on diag dominant");
    assert!(
        result.residual_norm < 1e-8,
        "Residual too large: {}",
        result.residual_norm
    );

    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

// ============================================================================
// SOR Tests
// ============================================================================

#[test]
fn test_sor_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .sor(
            &a,
            &b,
            None,
            SorOptions {
                max_iter: 500,
                omega: 1.5,
                rtol: 1e-10,
                atol: 1e-14,
            },
        )
        .expect("SOR should succeed");

    assert!(result.converged, "SOR should converge on Laplacian");
    assert!(
        result.residual_norm < 1e-8,
        "Residual too large: {}",
        result.residual_norm
    );

    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_sor_vs_jacobi() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_diag_dominant(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let jacobi_result = client
        .jacobi(
            &a,
            &b,
            None,
            JacobiOptions {
                max_iter: 500,
                omega: 2.0 / 3.0,
                ..Default::default()
            },
        )
        .expect("Jacobi should succeed");

    let sor_result = client
        .sor(
            &a,
            &b,
            None,
            SorOptions {
                max_iter: 500,
                omega: 1.2,
                ..Default::default()
            },
        )
        .expect("SOR should succeed");

    assert!(jacobi_result.converged);
    assert!(sor_result.converged);
    assert!(
        sor_result.iterations <= jacobi_result.iterations,
        "SOR ({}) should converge no slower than Jacobi ({})",
        sor_result.iterations,
        jacobi_result.iterations
    );
}

// ============================================================================
// LGMRES Tests
// ============================================================================

#[test]
fn test_lgmres_nonsymmetric() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_nonsymmetric(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .lgmres(
            &a,
            &b,
            None,
            LgmresOptions {
                max_iter: 100,
                restart: 10,
                k_aug: 3,
                rtol: 1e-10,
                atol: 1e-14,
                preconditioner: PreconditionerType::None,
            },
        )
        .expect("LGMRES should succeed");

    assert!(result.converged, "LGMRES should converge");
    assert!(
        result.residual_norm < 1e-8,
        "Residual: {}",
        result.residual_norm
    );

    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_lgmres_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .lgmres(&a, &b, None, LgmresOptions::default())
        .expect("LGMRES should succeed");

    assert!(result.converged, "LGMRES should converge on Laplacian");
}

// ============================================================================
// QMR Tests
// ============================================================================

#[test]
fn test_qmr_nonsymmetric() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_nonsymmetric(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .qmr(
            &a,
            &b,
            None,
            QmrOptions {
                max_iter: 200,
                rtol: 1e-10,
                atol: 1e-14,
                preconditioner: PreconditionerType::None,
            },
        )
        .expect("QMR should succeed");

    assert!(result.converged, "QMR should converge");
    assert!(
        result.residual_norm < 1e-6,
        "Residual: {}",
        result.residual_norm
    );

    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-6, "Verification residual: {}", res_norm);
}

#[test]
fn test_qmr_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let result = client
        .qmr(&a, &b, None, QmrOptions::default())
        .expect("QMR should succeed");

    assert!(result.converged, "QMR should converge on SPD Laplacian");
}

// ============================================================================
// AMG Tests
// ============================================================================

#[test]
fn test_amg_preconditioned_cg() {
    use numr::algorithm::iterative::{AmgOptions, amg_preconditioned_cg, amg_setup};

    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 20;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device);

    let hierarchy =
        amg_setup(&client, &a, AmgOptions::default()).expect("AMG setup should succeed");

    let (solution, iterations, residual_norm, converged) =
        amg_preconditioned_cg(&client, &a, &b, None, &hierarchy, 100, 1e-10, 1e-14)
            .expect("AMG-CG should succeed");

    assert!(converged, "AMG-preconditioned CG should converge");
    assert!(residual_norm < 1e-8, "Residual: {}", residual_norm);
    assert!(iterations > 0, "Should take at least 1 iteration");

    let ax = a.spmv(&solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}
