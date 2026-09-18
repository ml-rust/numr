// Backend parity tests for FwhtOps.
//
// This file checks CPU, the reference backend, against an explicit O(n^2)
// Sylvester-Hadamard matrix product. `cuda.rs` checks CUDA against CPU.
// `wgpu.rs` is a stub until that kernel lands.

pub mod cuda;
pub mod wgpu;

use numr::dtype::DType;
use numr::ops::FwhtOps;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::common::{assert_allclose_f64, create_cpu_client};

/// Sylvester-Hadamard reference matrix: `H[i][j] = (-1)^popcount(i & j) / sqrt(n)`.
fn sylvester_hadamard(n: usize) -> Vec<Vec<f64>> {
    let scale = 1.0 / (n as f64).sqrt();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if (i & j).count_ones() % 2 == 0 {
                        scale
                    } else {
                        -scale
                    }
                })
                .collect()
        })
        .collect()
}

/// Applies the reference transform to every `block_size` segment of every row.
fn reference_fwht(data: &[f64], last_dim: usize, block_size: usize) -> Vec<f64> {
    let h = sylvester_hadamard(block_size);
    let mut out = vec![0.0f64; data.len()];
    for (row_in, row_out) in data.chunks(last_dim).zip(out.chunks_mut(last_dim)) {
        for (block_in, block_out) in row_in
            .chunks(block_size)
            .zip(row_out.chunks_mut(block_size))
        {
            for i in 0..block_size {
                block_out[i] = (0..block_size).map(|j| h[i][j] * block_in[j]).sum();
            }
        }
    }
    out
}

fn test_fwht_dtype(dtype: DType, tol: f64) {
    // Two rows exercises the sequential path; 128 rows crosses the Rayon
    // row threshold, so both kernel paths are covered.
    test_fwht_shape(dtype, tol, [2, 2048], 1024);
    test_fwht_shape(dtype, tol, [128, 64], 32);
}

fn test_fwht_shape(dtype: DType, tol: f64, shape: [usize; 2], block_size: usize) {
    let (client, device) = create_cpu_client();

    let last_dim = shape[1];
    let numel: usize = shape.iter().product();

    let data: Vec<f64> = (0..numel)
        .map(|i| ((i % 97) as f64) * 0.031 - 1.5)
        .collect();

    let x = tensor_from_f64(&data, &shape, dtype, &device, &client).expect("tensor creation");
    let result = client.fwht(&x, block_size, None).expect("fwht failed");

    let got = result
        .to_dtype(DType::F64)
        .expect("cast to f64")
        .to_vec::<f64>();
    let expected = reference_fwht(&data, last_dim, block_size);

    assert_allclose_f64(
        &got,
        &expected,
        tol,
        tol,
        &format!("fwht_{dtype:?}_{shape:?}"),
    );
}

#[test]
fn test_fwht_f32() {
    test_fwht_dtype(DType::F32, 1e-3);
}

#[test]
fn test_fwht_f64() {
    test_fwht_dtype(DType::F64, 1e-9);
}

#[cfg(feature = "f16")]
#[test]
fn test_fwht_f16() {
    test_fwht_dtype(DType::F16, 0.2);
}

#[cfg(feature = "f16")]
#[test]
fn test_fwht_bf16() {
    test_fwht_dtype(DType::BF16, 1.0);
}
