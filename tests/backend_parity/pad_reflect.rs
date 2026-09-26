// Backend parity tests for reflect-mode pad (`ShapeOps::pad_mode` with
// `PadMode::Reflect`). CPU is the reference; CUDA and WebGPU must match it.

use numr::dtype::DType;
use numr::ops::{PadMode, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Small positive integers: exact in every dtype the parity sweep carries,
/// and distinct enough along a row that a shifted column is caught.
fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i % 13) as f64 + 1.0).collect()
}

fn narrow_all<R: Runtime>(t: &Tensor<R>, dims: &[(isize, usize, usize)]) -> Tensor<R> {
    let mut out = t.clone();
    for &(dim, start, len) in dims {
        out = out.narrow(dim, start, len).unwrap();
    }
    out
}

fn check_pad_reflect(
    shape: &[usize],
    narrow: &[(isize, usize, usize)],
    padding: &[usize],
    dtype: DType,
) {
    let data = ramp(shape.iter().product());
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(&data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_view = narrow_all(&cpu_tensor, narrow);
    let cpu_result = cpu_client
        .pad_mode(&cpu_view, padding, PadMode::Reflect)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let tensor = tensor_from_f64(&data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let view = narrow_all(&tensor, narrow);
            let result = cuda_client
                .pad_mode(&view, padding, PadMode::Reflect)
                .unwrap();
            assert_eq!(cpu_result.shape(), result.shape());
            assert_tensor_allclose(
                &result,
                &cpu_result,
                dtype,
                &format!(
                    "pad_reflect CUDA vs CPU shape={shape:?} narrow={narrow:?} pad={padding:?}"
                ),
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let tensor = tensor_from_f64(&data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let view = narrow_all(&tensor, narrow);
            let result = wgpu_client
                .pad_mode(&view, padding, PadMode::Reflect)
                .unwrap();
            assert_eq!(cpu_result.shape(), result.shape());
            assert_tensor_allclose(
                &result,
                &cpu_result,
                dtype,
                &format!(
                    "pad_reflect WebGPU vs CPU shape={shape:?} narrow={narrow:?} pad={padding:?}"
                ),
            );
        });
    }
}

fn all_dtypes() -> Vec<DType> {
    parity_dtypes(DTypeDomain::AllNumeric, "cpu")
}

#[test]
fn test_pad_reflect_last_dim() {
    for dtype in all_dtypes() {
        check_pad_reflect(&[3, 8], &[], &[2, 3], dtype);
        check_pad_reflect(&[3, 8], &[], &[0, 5], dtype);
    }
}

#[test]
fn test_pad_reflect_second_last_dim() {
    for dtype in all_dtypes() {
        check_pad_reflect(&[3, 8], &[], &[0, 0, 1, 2], dtype);
        check_pad_reflect(&[8, 3], &[], &[0, 0, 3, 4], dtype);
    }
}

#[test]
fn test_pad_reflect_both_trailing_dims() {
    for dtype in all_dtypes() {
        check_pad_reflect(&[8, 8], &[], &[1, 2, 2, 1], dtype);
    }
}

#[test]
fn test_pad_reflect_rank1() {
    for dtype in all_dtypes() {
        check_pad_reflect(&[11], &[], &[3, 2], dtype);
        check_pad_reflect(&[32], &[], &[16, 16], dtype);
    }
}

#[test]
fn test_pad_reflect_rank3_leading_batch() {
    for dtype in all_dtypes() {
        check_pad_reflect(&[2, 3, 8], &[], &[1, 1], dtype);
        check_pad_reflect(&[2, 3, 8], &[], &[0, 0, 1, 0], dtype);
    }
}

#[test]
fn test_pad_reflect_leading_dim_generic_path() {
    // Padding a leading dimension exercises the same generic decode as a
    // trailing-dim pad; there is no row-optimized fast path for reflect.
    for dtype in all_dtypes() {
        check_pad_reflect(&[2, 3, 4, 5], &[], &[0, 0, 0, 0, 1, 0], dtype);
        check_pad_reflect(&[2, 3, 4, 5], &[], &[1, 1, 0, 0, 0, 1, 1, 0], dtype);
    }
}

#[test]
fn test_pad_reflect_zero_pads() {
    for dtype in all_dtypes() {
        check_pad_reflect(&[3, 8], &[], &[0, 0], dtype);
        check_pad_reflect(&[3, 8], &[], &[0, 0, 0, 0], dtype);
    }
}

#[test]
fn test_pad_reflect_non_contiguous_view() {
    // `narrow` on the last dim keeps the view contiguous with a moved base
    // pointer, exercising the `ensure_contiguous` / clone path uniformly.
    for dtype in all_dtypes() {
        check_pad_reflect(&[40], &[(0, 1, 32)], &[0, 16], dtype);
        check_pad_reflect(&[4, 33], &[(1, 1, 32)], &[0, 16], dtype);
    }
}

#[test]
fn test_pad_reflect_rejects_pad_at_least_dim_size() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let data = ramp(4);
    let tensor = tensor_from_f64(&data, &[4], DType::F32, &cpu_device, &cpu_client).unwrap();
    assert!(
        cpu_client
            .pad_mode(&tensor, &[4, 0], PadMode::Reflect)
            .is_err()
    );
    assert!(
        cpu_client
            .pad_mode(&tensor, &[0, 5], PadMode::Reflect)
            .is_err()
    );
}
