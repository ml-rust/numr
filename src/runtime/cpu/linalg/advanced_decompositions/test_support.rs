#![cfg(test)]
//! Shared test-only helpers for `runtime::cpu::linalg::advanced_decompositions` unit tests.

use super::super::super::{CpuClient, CpuDevice};

pub(super) fn create_client() -> CpuClient {
    let device = CpuDevice::new();
    CpuClient::new(device)
}

pub(super) fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() < tol,
        "Expected {} to be close to {}, diff = {}",
        a,
        b,
        (a - b).abs()
    );
}

pub(super) fn matrix_multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    c
}

pub(super) fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

pub(super) fn is_orthogonal(q: &[f64], n: usize, tol: f64) -> bool {
    let qt = transpose(q, n);
    let qtq = matrix_multiply(&qt, q, n);
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (qtq[i * n + j] - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}
