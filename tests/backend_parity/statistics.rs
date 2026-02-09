// Backend parity tests for StatisticalOps.
//
// These tests enforce parity + correctness: each backend result must match
// expected behavior and stay aligned with CPU semantics.

use numr::ops::StatisticalOps;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn test_cov_basic_parity() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(&[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2], &$device);
            let cov = $client.cov(&a, None).unwrap();
            assert_eq!(cov.shape(), &[2, 2], "cov shape mismatch on {}", $backend);
            let data: Vec<f32> = cov.to_vec();
            assert!(
                approx_eq(data[0], 1.0, 1e-5),
                "cov[0,0] mismatch on {}",
                $backend
            );
            assert!(
                approx_eq(data[1], 1.0, 1e-5),
                "cov[0,1] mismatch on {}",
                $backend
            );
            assert!(
                approx_eq(data[2], 1.0, 1e-5),
                "cov[1,0] mismatch on {}",
                $backend
            );
            assert!(
                approx_eq(data[3], 1.0, 1e-5),
                "cov[1,1] mismatch on {}",
                $backend
            );
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_corrcoef_range_parity() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(
                &[
                    1.0f32, 5.0, 2.0, 3.0, 4.0, 1.0, 5.0, 2.0, 3.0, 4.0, 6.0, 7.0,
                ],
                &[4, 3],
                &$device,
            );
            let corr = $client.corrcoef(&a).unwrap();
            let data: Vec<f32> = corr.to_vec();
            for (i, &v) in data.iter().enumerate() {
                assert!(
                    (-1.0 - 1e-5..=1.0 + 1e-5).contains(&v),
                    "corr[{}]={} out of range on {}",
                    i,
                    v,
                    $backend
                );
            }
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_skew_kurtosis_parity() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let sym = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &$device);
            let skew = $client.skew(&sym, &[], false, 0).unwrap();
            let skew_data: Vec<f32> = skew.to_vec();
            assert!(
                skew_data[0].abs() < 0.1,
                "symmetric skew mismatch on {}: {}",
                $backend,
                skew_data[0]
            );

            let heavy = Tensor::from_slice(
                &[-100.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                &[10],
                &$device,
            );
            let kurt = $client.kurtosis(&heavy, &[], false, 0).unwrap();
            let kurt_data: Vec<f32> = kurt.to_vec();
            assert!(
                kurt_data[0] > 0.0,
                "heavy-tail kurtosis mismatch on {}: {}",
                $backend,
                kurt_data[0]
            );
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_mode_parity_f32() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(&[1.0f32, 2.0, 2.0, 2.0, 3.0], &[5], &$device);
            let (values, counts) = $client.mode(&a, Some(0), false).unwrap();
            let values_data: Vec<f32> = values.to_vec();
            let counts_data: Vec<i64> = counts.to_vec();
            assert!(
                approx_eq(values_data[0], 2.0, 1e-5),
                "mode value mismatch on {}",
                $backend
            );
            assert_eq!(counts_data[0], 3, "mode count mismatch on {}", $backend);
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_mode_parity_i32() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(&[1i32, 2, 2, 3, 2], &[5], &$device);
            let (values, counts) = $client.mode(&a, Some(0), false).unwrap();
            let values_data: Vec<i32> = values.to_vec();
            let counts_data: Vec<i64> = counts.to_vec();
            assert_eq!(values_data[0], 2, "mode i32 value mismatch on {}", $backend);
            assert_eq!(counts_data[0], 3, "mode i32 count mismatch on {}", $backend);
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_quantile_percentile_median_parity() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &$device);

            let q = $client.quantile(&a, 0.5, Some(0), false, "linear").unwrap();
            let q_data: Vec<f32> = q.to_vec();
            assert!(
                approx_eq(q_data[0], 2.5, 1e-5),
                "quantile mismatch on {}: {}",
                $backend,
                q_data[0]
            );

            let p = $client.percentile(&a, 50.0, Some(0), false).unwrap();
            let p_data: Vec<f32> = p.to_vec();
            assert!(
                approx_eq(p_data[0], 2.5, 1e-5),
                "percentile mismatch on {}: {}",
                $backend,
                p_data[0]
            );

            let m = $client.median(&a, Some(0), false).unwrap();
            let m_data: Vec<f32> = m.to_vec();
            assert!(
                approx_eq(m_data[0], 2.5, 1e-5),
                "median mismatch on {}: {}",
                $backend,
                m_data[0]
            );
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_quantile_invalid_inputs_parity() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &$device);
            assert!(
                $client
                    .quantile(&a, -0.1, Some(0), false, "linear")
                    .is_err(),
                "quantile q<0 should error on {}",
                $backend
            );
            assert!(
                $client.quantile(&a, 1.1, Some(0), false, "linear").is_err(),
                "quantile q>1 should error on {}",
                $backend
            );
            assert!(
                $client.percentile(&a, -1.0, Some(0), false).is_err(),
                "percentile p<0 should error on {}",
                $backend
            );
            assert!(
                $client.percentile(&a, 101.0, Some(0), false).is_err(),
                "percentile p>100 should error on {}",
                $backend
            );
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_quantile_f64_parity() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &$device);
            let q = $client.quantile(&a, 0.5, Some(0), false, "linear").unwrap();
            let q_data: Vec<f64> = q.to_vec();
            assert!(
                approx_eq_f64(q_data[0], 3.0, 1e-10),
                "f64 quantile mismatch on {}: {}",
                $backend,
                q_data[0]
            );
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_histogram_parity() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(&[0.5f32, 1.5, 2.5, 3.5, 4.5], &[5], &$device);
            let (hist, edges) = $client.histogram(&a, 5, Some((0.0, 5.0))).unwrap();
            assert_eq!(hist.shape(), &[5], "hist shape mismatch on {}", $backend);
            assert_eq!(edges.shape(), &[6], "edges shape mismatch on {}", $backend);
            let hist_data: Vec<i64> = hist.to_vec();
            assert_eq!(
                hist_data,
                vec![1, 1, 1, 1, 1],
                "hist counts mismatch on {}",
                $backend
            );
            let edges_data: Vec<f32> = edges.to_vec();
            assert!(
                approx_eq(edges_data[0], 0.0, 1e-5) && approx_eq(edges_data[5], 5.0, 1e-5),
                "hist edges mismatch on {}",
                $backend
            );
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}

#[test]
fn test_histogram_invalid_inputs_parity() {
    macro_rules! run {
        ($client:expr, $device:expr, $backend:expr) => {{
            let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &$device);
            assert!(
                $client.histogram(&a, 0, None).is_err(),
                "hist bins=0 should error on {}",
                $backend
            );
            assert!(
                $client.histogram(&a, 5, Some((5.0, 5.0))).is_err(),
                "hist invalid range should error on {}",
                $backend
            );
            assert!(
                $client.histogram(&a, 5, Some((10.0, 5.0))).is_err(),
                "hist invalid descending range should error on {}",
                $backend
            );
        }};
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    run!(cpu_client, cpu_device, "cpu");
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        run!(cuda_client, cuda_device, "cuda");
    });
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        run!(wgpu_client, wgpu_device, "wgpu");
    });
}
