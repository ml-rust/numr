// Backend parity tests for pad layouts that reach the CUDA row-wise kernels.
//
// The CUDA launcher routes a pad whose leading dimensions keep their extent to
// `pad_rows_<dtype>` or, when every row start on both sides is 16-byte
// aligned, to `pad_rows_vec_<dtype>`. Every other layout stays on the generic
// per-element kernel. Each case here is checked against CPU `pad`, and the
// generic path is covered alongside, so the three implementations are tied to
// one reference.

use numr::dtype::DType;
use numr::ops::ShapeOps;
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

fn check_pad(
    shape: &[usize],
    narrow: &[(isize, usize, usize)],
    padding: &[usize],
    value: f64,
    dtype: DType,
) {
    let data = ramp(shape.iter().product());
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(&data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_view = narrow_all(&cpu_tensor, narrow);
    let cpu_result = cpu_client.pad(&cpu_view, padding, value).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let tensor = tensor_from_f64(&data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let view = narrow_all(&tensor, narrow);
            let result = cuda_client.pad(&view, padding, value).unwrap();
            assert_eq!(cpu_result.shape(), result.shape());
            assert_tensor_allclose(
                &result,
                &cpu_result,
                dtype,
                &format!("pad CUDA vs CPU shape={shape:?} narrow={narrow:?} pad={padding:?}"),
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let tensor = tensor_from_f64(&data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let view = narrow_all(&tensor, narrow);
            let result = wgpu_client.pad(&view, padding, value).unwrap();
            assert_eq!(cpu_result.shape(), result.shape());
            assert_tensor_allclose(
                &result,
                &cpu_result,
                dtype,
                &format!("pad WebGPU vs CPU shape={shape:?} narrow={narrow:?} pad={padding:?}"),
            );
        });
    }
}

fn all_dtypes() -> Vec<DType> {
    parity_dtypes(DTypeDomain::AllNumeric, "cpu")
}

#[test]
fn test_pad_rows_last_dim() {
    for dtype in all_dtypes() {
        check_pad(&[3, 8], &[], &[2, 3], 0.0, dtype);
        check_pad(&[3, 8], &[], &[0, 5], 7.0, dtype);
    }
}

#[test]
fn test_pad_rows_second_last_dim() {
    for dtype in all_dtypes() {
        check_pad(&[3, 8], &[], &[0, 0, 1, 2], 0.0, dtype);
        check_pad(&[3, 8], &[], &[0, 0, 0, 3], 2.0, dtype);
    }
}

#[test]
fn test_pad_rows_both_trailing_dims() {
    for dtype in all_dtypes() {
        check_pad(&[3, 8], &[], &[1, 2, 2, 1], 0.0, dtype);
    }
}

#[test]
fn test_pad_rows_rank1() {
    for dtype in all_dtypes() {
        check_pad(&[11], &[], &[3, 2], 0.0, dtype);
        check_pad(&[32], &[], &[16, 16], 1.0, dtype);
    }
}

#[test]
fn test_pad_rows_rank3_leading_batch() {
    for dtype in all_dtypes() {
        check_pad(&[2, 3, 8], &[], &[1, 1], 0.0, dtype);
        check_pad(&[2, 3, 8], &[], &[0, 0, 1, 0], 0.0, dtype);
        check_pad(&[2, 3, 16], &[], &[16, 16, 1, 1], 4.0, dtype);
    }
}

#[test]
fn test_pad_rows_vector_aligned_widths() {
    // Source, output, and pad_before byte widths are all 16-byte multiples for
    // every element size from 1 to 16 bytes, so the launcher picks the
    // 128-bit chunk kernel.
    for dtype in all_dtypes() {
        check_pad(&[4, 16], &[], &[16, 16], 0.0, dtype);
        check_pad(&[5, 16], &[], &[0, 0, 2, 3], 3.0, dtype);
        check_pad(&[2, 4, 32], &[], &[0, 16, 1, 0], 0.0, dtype);
    }
}

#[test]
fn test_pad_rows_odd_column_count() {
    // A 7-wide row is never a 16-byte multiple, so the scalar row kernel runs.
    for dtype in all_dtypes() {
        check_pad(&[4, 7], &[], &[0, 9], 0.0, dtype);
        check_pad(&[4, 7], &[], &[3, 0, 1, 1], 5.0, dtype);
    }
}

#[test]
fn test_pad_rows_unaligned_view() {
    // `narrow` on the last dim of a rank-1 tensor keeps the view contiguous
    // and moves its base pointer by one element, so the row widths pass the
    // vector guard while the base alignment fails it.
    for dtype in all_dtypes() {
        check_pad(&[40], &[(0, 1, 32)], &[0, 16], 0.0, dtype);
        check_pad(&[40], &[(0, 1, 32)], &[16, 0], 0.0, dtype);
    }
}

#[test]
fn test_pad_rows_narrow_last_dim_matrix() {
    // Non-contiguous view: the backend copies it before padding.
    for dtype in all_dtypes() {
        check_pad(&[4, 33], &[(1, 1, 32)], &[0, 16], 0.0, dtype);
        check_pad(&[4, 33], &[(1, 1, 16)], &[0, 0, 1, 1], 0.0, dtype);
    }
}

#[test]
fn test_pad_rows_zero_pads() {
    for dtype in all_dtypes() {
        check_pad(&[3, 8], &[], &[0, 0], 0.0, dtype);
        check_pad(&[3, 8], &[], &[0, 0, 0, 0], 0.0, dtype);
        check_pad(&[3, 8], &[], &[0, 3], 0.0, dtype);
        check_pad(&[3, 8], &[], &[0, 0, 2, 0], 0.0, dtype);
    }
}

#[test]
fn test_pad_rows_grid_stride_rows_and_batch() {
    // Row and batch counts above the 65535 y/z grid limit, so the kernels
    // grid-stride on those axes.
    for dtype in [DType::F32, DType::I32, DType::U8] {
        check_pad(&[70_000, 3], &[], &[1, 0], 0.0, dtype);
        check_pad(&[70_000, 1, 3], &[], &[0, 1], 0.0, dtype);
    }
}

#[test]
fn test_pad_generic_path_leading_dims() {
    // Padding a leading dimension keeps the generic per-element kernel.
    for dtype in all_dtypes() {
        check_pad(&[2, 3, 4, 5], &[], &[0, 0, 0, 0, 1, 0], 0.0, dtype);
        check_pad(&[2, 3, 8], &[], &[0, 0, 0, 0, 1, 1], 0.0, dtype);
        check_pad(&[2, 3, 4, 5], &[], &[1, 1, 0, 0, 0, 1, 1, 0], 6.0, dtype);
    }
}
