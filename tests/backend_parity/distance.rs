// Backend parity tests for DistanceOps trait
//
// Tests: cdist, pdist, squareform, squareform_inverse
// CPU is the reference implementation; CUDA and WebGPU must match.

use numr::dtype::DType;
use numr::ops::{DistanceMetric, DistanceOps};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

// ============================================================================
// cdist
// ============================================================================

struct CdistCase {
    x: Vec<f64>,
    x_shape: Vec<usize>,
    y: Vec<f64>,
    y_shape: Vec<usize>,
    metric: DistanceMetric,
}

impl CdistCase {
    fn new(
        x: Vec<f64>,
        x_shape: Vec<usize>,
        y: Vec<f64>,
        y_shape: Vec<usize>,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            x,
            x_shape,
            y,
            y_shape,
            metric,
        }
    }
}

fn cdist_test_cases() -> Vec<CdistCase> {
    // Points in 2D
    let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]; // 3 points in 2D
    let y = vec![1.0, 1.0, 2.0, 0.0]; // 2 points in 2D

    vec![
        CdistCase::new(
            x.clone(),
            vec![3, 2],
            y.clone(),
            vec![2, 2],
            DistanceMetric::Euclidean,
        ),
        CdistCase::new(
            x.clone(),
            vec![3, 2],
            y.clone(),
            vec![2, 2],
            DistanceMetric::SquaredEuclidean,
        ),
        CdistCase::new(
            x.clone(),
            vec![3, 2],
            y.clone(),
            vec![2, 2],
            DistanceMetric::Manhattan,
        ),
        CdistCase::new(
            x.clone(),
            vec![3, 2],
            y.clone(),
            vec![2, 2],
            DistanceMetric::Chebyshev,
        ),
        // 3D points
        CdistCase::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            vec![3, 3],
            DistanceMetric::Euclidean,
        ),
    ]
}

fn test_cdist_parity(dtype: DType) {
    let cases = cdist_test_cases();
    let (cpu_client, cpu_device) = create_cpu_client();

    for (idx, tc) in cases.iter().enumerate() {
        let cpu_x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU x tensor failed");
        let cpu_y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU y tensor failed");
        let cpu_result = cpu_client
            .cdist(&cpu_x, &cpu_y, tc.metric)
            .unwrap_or_else(|e| panic!("CPU cdist {:?} failed for {dtype:?}: {e}", tc.metric));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &cuda_device, &cuda_client)
                    .expect("CUDA x tensor failed");
                let y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &cuda_device, &cuda_client)
                    .expect("CUDA y tensor failed");
                let result = cuda_client
                    .cdist(&x, &y, tc.metric)
                    .unwrap_or_else(|e| panic!("CUDA cdist failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("cdist {:?} CUDA vs CPU [{dtype:?}] case {idx}", tc.metric),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("WebGPU x tensor failed");
                let y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("WebGPU y tensor failed");
                let result = wgpu_client
                    .cdist(&x, &y, tc.metric)
                    .unwrap_or_else(|e| panic!("WebGPU cdist failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("cdist {:?} WebGPU vs CPU [{dtype:?}] case {idx}", tc.metric),
                );
            });
        }
    }
}

#[test]
fn test_cdist_parity_all_dtypes() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_cdist_parity(dtype);
    }
}

// ============================================================================
// pdist
// ============================================================================

fn test_pdist_parity(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    // 4 points in 2D
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let shape = vec![4, 2];

    let metrics = vec![
        DistanceMetric::Euclidean,
        DistanceMetric::SquaredEuclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Chebyshev,
    ];

    for metric in &metrics {
        let cpu_x = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU tensor failed");
        let cpu_result = cpu_client
            .pdist(&cpu_x, *metric)
            .unwrap_or_else(|e| panic!("CPU pdist {metric:?} failed: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .expect("CUDA tensor failed");
                let result = cuda_client
                    .pdist(&x, *metric)
                    .unwrap_or_else(|e| panic!("CUDA pdist failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("pdist {metric:?} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("WebGPU tensor failed");
                let result = wgpu_client
                    .pdist(&x, *metric)
                    .unwrap_or_else(|e| panic!("WebGPU pdist failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("pdist {metric:?} WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_pdist_parity_all_dtypes() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_pdist_parity(dtype);
    }
}

// ============================================================================
// squareform roundtrip
// ============================================================================

#[test]
fn test_squareform_roundtrip_parity() {
    let dtype = DType::F32;
    let (cpu_client, cpu_device) = create_cpu_client();

    // 4 points in 2D
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let shape = vec![4, 2];
    let n = 4usize;

    let cpu_x =
        tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client).expect("tensor failed");
    let cpu_condensed = cpu_client
        .pdist(&cpu_x, DistanceMetric::Euclidean)
        .expect("pdist failed");
    let cpu_square = cpu_client
        .squareform(&cpu_condensed, n)
        .expect("squareform failed");
    let cpu_back = cpu_client
        .squareform_inverse(&cpu_square)
        .expect("squareform_inverse failed");

    // Verify roundtrip: condensed -> square -> condensed
    assert_tensor_allclose(&cpu_back, &cpu_condensed, dtype, "squareform roundtrip CPU");

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
            .expect("tensor failed");
        let condensed = wgpu_client
            .pdist(&x, DistanceMetric::Euclidean)
            .expect("pdist failed");
        let square = wgpu_client
            .squareform(&condensed, n)
            .expect("squareform failed");

        assert_tensor_allclose(&square, &cpu_square, dtype, "squareform WebGPU vs CPU");

        let back = wgpu_client
            .squareform_inverse(&square)
            .expect("squareform_inverse failed");
        assert_tensor_allclose(
            &back,
            &cpu_condensed,
            dtype,
            "squareform_inverse WebGPU vs CPU",
        );
    });
}

// ============================================================================
// cosine distance
// ============================================================================

#[test]
fn test_cdist_cosine_parity() {
    let dtype = DType::F32;
    let (cpu_client, cpu_device) = create_cpu_client();

    let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 points in 2D
    let y = vec![1.0, 0.0, 0.0, 1.0]; // 2 points in 2D

    let cpu_x =
        tensor_from_f64(&x, &[3, 2], dtype, &cpu_device, &cpu_client).expect("tensor failed");
    let cpu_y =
        tensor_from_f64(&y, &[2, 2], dtype, &cpu_device, &cpu_client).expect("tensor failed");
    let _cpu_result = cpu_client
        .cdist(&cpu_x, &cpu_y, DistanceMetric::Cosine)
        .expect("CPU cosine cdist failed");

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wx =
            tensor_from_f64(&x, &[3, 2], dtype, &wgpu_device, &wgpu_client).expect("tensor failed");
        let wy =
            tensor_from_f64(&y, &[2, 2], dtype, &wgpu_device, &wgpu_client).expect("tensor failed");
        let result = wgpu_client
            .cdist(&wx, &wy, DistanceMetric::Cosine)
            .expect("WebGPU cosine cdist failed");
        assert_tensor_allclose(&result, &_cpu_result, dtype, "cdist Cosine WebGPU vs CPU");
    });
}

// ============================================================================
// cosine and correlation at extreme magnitudes
//
// The denominator is `sqrt(norm_a) * sqrt(norm_b)`. The old `sqrt(norm_a *
// norm_b)` overflowed for large rows and flushed to zero for tiny rows.
// ============================================================================

const EXTREME_COLS: usize = 8;

fn extreme_rows() -> Vec<f64> {
    vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
        8.0, 1.5, 6.0, 2.5, 4.0, 3.5, 2.0, 9.0, //
        -3.0, 5.0, 0.5, -2.0, 7.0, 1.0, -4.0, 6.0,
    ]
}

fn extreme_other_rows() -> Vec<f64> {
    vec![
        2.0, 1.0, 4.0, 3.0, 8.0, 5.0, 6.0, 7.0, //
        -1.0, 3.0, 2.0, 5.0, 1.0, 4.0, 9.0, 0.5,
    ]
}

/// Reference distance in f64: `1 - dot / (|a| |b|)`, or Pearson for `Correlation`.
fn analytic_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    let n = a.len() as f64;
    let (a, b): (Vec<f64>, Vec<f64>) = match metric {
        DistanceMetric::Correlation => {
            let ma = a.iter().sum::<f64>() / n;
            let mb = b.iter().sum::<f64>() / n;
            (
                a.iter().map(|v| v - ma).collect(),
                b.iter().map(|v| v - mb).collect(),
            )
        }
        _ => (a.to_vec(), b.to_vec()),
    };
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    1.0 - dot / (na * nb)
}

fn extreme_cdist_parity(metric: DistanceMetric, scale: f64) {
    let dtype = DType::F32;
    let x: Vec<f64> = extreme_rows().iter().map(|v| v * scale).collect();
    let y: Vec<f64> = extreme_other_rows().iter().map(|v| v * scale).collect();
    let (nx, ny) = (x.len() / EXTREME_COLS, y.len() / EXTREME_COLS);
    let label = format!("cdist {metric:?} scale {scale:e}");

    // Unit-scale f64 rows give the reference. Scaling cannot change the distance.
    let base_x = extreme_rows();
    let base_y = extreme_other_rows();
    let mut want = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            let a = &base_x[i * EXTREME_COLS..(i + 1) * EXTREME_COLS];
            let b = &base_y[j * EXTREME_COLS..(j + 1) * EXTREME_COLS];
            want.push(analytic_distance(a, b, metric));
        }
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_x = tensor_from_f64(&x, &[nx, EXTREME_COLS], dtype, &cpu_device, &cpu_client)
        .expect("CPU x tensor failed");
    let cpu_y = tensor_from_f64(&y, &[ny, EXTREME_COLS], dtype, &cpu_device, &cpu_client)
        .expect("CPU y tensor failed");
    let cpu_result = cpu_client
        .cdist(&cpu_x, &cpu_y, metric)
        .unwrap_or_else(|e| panic!("CPU {label} failed: {e}"));

    let got: Vec<f32> = cpu_result.to_vec();
    assert_eq!(got.len(), want.len(), "{label}: CPU result length");
    for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (*g as f64 - w).abs() <= 1e-5,
            "{label}: CPU element {idx} is {g:e}, analytic reference is {w:e}"
        );
    }

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cx = tensor_from_f64(&x, &[nx, EXTREME_COLS], dtype, &cuda_device, &cuda_client)
                .expect("CUDA x tensor failed");
            let cy = tensor_from_f64(&y, &[ny, EXTREME_COLS], dtype, &cuda_device, &cuda_client)
                .expect("CUDA y tensor failed");
            let result = cuda_client
                .cdist(&cx, &cy, metric)
                .unwrap_or_else(|e| panic!("CUDA {label} failed: {e}"));
            assert_tensor_allclose(&result, &cpu_result, dtype, &format!("{label} CUDA vs CPU"));
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wx = tensor_from_f64(&x, &[nx, EXTREME_COLS], dtype, &wgpu_device, &wgpu_client)
                .expect("WebGPU x tensor failed");
            let wy = tensor_from_f64(&y, &[ny, EXTREME_COLS], dtype, &wgpu_device, &wgpu_client)
                .expect("WebGPU y tensor failed");
            let result = wgpu_client
                .cdist(&wx, &wy, metric)
                .unwrap_or_else(|e| panic!("WebGPU {label} failed: {e}"));
            assert_tensor_allclose(
                &result,
                &cpu_result,
                dtype,
                &format!("{label} WebGPU vs CPU"),
            );
        });
    }
}

#[test]
fn test_cdist_cosine_huge_magnitude_parity() {
    extreme_cdist_parity(DistanceMetric::Cosine, 1e10);
}

#[test]
fn test_cdist_cosine_tiny_magnitude_parity() {
    extreme_cdist_parity(DistanceMetric::Cosine, 1e-12);
}

#[test]
fn test_cdist_correlation_huge_magnitude_parity() {
    extreme_cdist_parity(DistanceMetric::Correlation, 1e10);
}

#[test]
fn test_cdist_correlation_tiny_magnitude_parity() {
    extreme_cdist_parity(DistanceMetric::Correlation, 1e-12);
}
