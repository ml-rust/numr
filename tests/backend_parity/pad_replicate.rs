// Backend parity tests for replicate-mode pad (`ShapeOps::pad_mode` with
// `PadMode::Replicate`). Every backend is compared against a hand-computed
// expectation, not against another backend.

use numr::dtype::DType;
use numr::ops::{PadMode, ShapeOps};
use numr::runtime::cpu::CpuRuntime;
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
/// and distinct along a row so a shifted column is caught.
fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i % 13) as f64 + 1.0).collect()
}

/// Reference implementation of `src = clamp(i - before, 0, size - 1)` per dim.
/// `padding` uses the PyTorch convention: pairs starting from the last dim.
fn reference(data: &[f64], shape: &[usize], padding: &[usize]) -> (Vec<usize>, Vec<f64>) {
    let ndim = shape.len();
    let mut before = vec![0usize; ndim];
    let mut after = vec![0usize; ndim];
    for (i, pair) in padding.chunks(2).enumerate() {
        before[ndim - 1 - i] = pair[0];
        after[ndim - 1 - i] = pair[1];
    }
    let out_shape: Vec<usize> = (0..ndim).map(|d| shape[d] + before[d] + after[d]).collect();
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let mut rem = idx;
        let mut coords = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        let mut src = 0usize;
        let mut stride = 1usize;
        for d in (0..ndim).rev() {
            let c = coords[d].saturating_sub(before[d]).min(shape[d] - 1);
            src += c * stride;
            stride *= shape[d];
        }
        out.push(data[src]);
    }
    (out_shape, out)
}

/// Run `pad_mode(Replicate)` on every enabled backend and compare with `expected`.
fn check_expected(
    shape: &[usize],
    padding: &[usize],
    out_shape: &[usize],
    expected: &[f64],
    dtype: DType,
) {
    let data = ramp(shape.iter().product());
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(&data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_client
        .pad_mode(&cpu_tensor, padding, PadMode::Replicate)
        .unwrap();
    assert_eq!(cpu_result.shape(), out_shape);
    let cpu_expected = tensor_from_f64(expected, out_shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU expected tensor failed for {dtype:?}: {e}"));
    assert_tensor_allclose(
        &cpu_result,
        &cpu_expected,
        dtype,
        &format!("pad_replicate CPU shape={shape:?} pad={padding:?}"),
    );

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|client, device| {
            let tensor = tensor_from_f64(&data, shape, dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let result = client
                .pad_mode(&tensor, padding, PadMode::Replicate)
                .unwrap();
            assert_eq!(result.shape(), out_shape);
            let want = tensor_from_f64(expected, out_shape, dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CUDA expected tensor failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &result,
                &want,
                dtype,
                &format!("pad_replicate CUDA shape={shape:?} pad={padding:?}"),
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|client, device| {
            let tensor = tensor_from_f64(&data, shape, dtype, &device, &client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let result = client
                .pad_mode(&tensor, padding, PadMode::Replicate)
                .unwrap();
            assert_eq!(result.shape(), out_shape);
            let want = tensor_from_f64(expected, out_shape, dtype, &device, &client)
                .unwrap_or_else(|e| panic!("WebGPU expected tensor failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &result,
                &want,
                dtype,
                &format!("pad_replicate WebGPU shape={shape:?} pad={padding:?}"),
            );
        });
    }
}

/// Check a case against the reference implementation.
fn check_reference(shape: &[usize], padding: &[usize], dtype: DType) {
    let data = ramp(shape.iter().product());
    let (out_shape, expected) = reference(&data, shape, padding);
    check_expected(shape, padding, &out_shape, &expected, dtype);
}

fn all_dtypes() -> Vec<DType> {
    parity_dtypes(DTypeDomain::AllNumeric, "cpu")
}

#[test]
fn test_reference_matches_hand_computed() {
    let (shape, out) = reference(&[1.0, 2.0, 3.0], &[3], &[2, 1]);
    assert_eq!(shape, [6]);
    assert_eq!(out, [1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_pad_replicate_1d_asymmetric() {
    for dtype in all_dtypes() {
        // [1, 2, 3] padded (2 before, 1 after).
        check_expected(&[3], &[2, 1], &[6], &[1., 1., 1., 2., 3., 3.], dtype);
        check_expected(&[3], &[0, 3], &[6], &[1., 2., 3., 3., 3., 3.], dtype);
    }
}

#[test]
fn test_pad_replicate_1d_pad_larger_than_input() {
    for dtype in all_dtypes() {
        // Size 2, pad 5 on each side: six copies of each edge.
        let mut expected = vec![1.0; 6];
        expected.extend([2.0; 6]);
        check_expected(&[2], &[5, 5], &[12], &expected, dtype);
        // Size 1: every output element is the single input element.
        check_expected(&[1], &[4, 3], &[8], &[1.0; 8], dtype);
    }
}

#[test]
fn test_pad_replicate_2d_asymmetric() {
    for dtype in all_dtypes() {
        // [[1,2,3],[4,5,6]], W pad (1, 2), H pad (2, 0).
        let r0 = [1., 1., 2., 3., 3., 3.];
        let r1 = [4., 4., 5., 6., 6., 6.];
        let expected: Vec<f64> = [r0, r0, r0, r1].concat();
        check_expected(&[2, 3], &[1, 2, 2, 0], &[4, 6], &expected, dtype);
    }
}

#[test]
fn test_pad_replicate_2d_zero_pad_on_some_dims() {
    for dtype in all_dtypes() {
        // Last dim unpadded, first dim (1, 1).
        let expected = [1., 2., 3., 1., 2., 3., 4., 5., 6., 4., 5., 6.];
        check_expected(&[2, 3], &[0, 0, 1, 1], &[4, 3], &expected, dtype);
        // First dim unpadded, last dim (0, 2).
        let expected = [1., 2., 3., 3., 3., 4., 5., 6., 6., 6.];
        check_expected(&[2, 3], &[0, 2], &[2, 5], &expected, dtype);
    }
}

#[test]
fn test_pad_replicate_4d_nchw() {
    for dtype in all_dtypes() {
        // [1, 2, 2, 2] with W pad (1, 0) and H pad (0, 1).
        // Channel 0 = [[1,2],[3,4]], channel 1 = [[5,6],[7,8]].
        let expected = [
            1., 1., 2., 3., 3., 4., 3., 3., 4., //
            5., 5., 6., 7., 7., 8., 7., 7., 8.,
        ];
        check_expected(
            &[1, 2, 2, 2],
            &[1, 0, 0, 1],
            &[1, 2, 3, 3],
            &expected,
            dtype,
        );
    }
}

#[test]
fn test_pad_replicate_4d_nchw_symmetric_and_large() {
    for dtype in all_dtypes() {
        check_reference(&[2, 3, 4, 5], &[2, 2, 1, 1], dtype);
        // Pad larger than both spatial dims, with a batch and channel dim untouched.
        check_reference(&[2, 3, 2, 3], &[7, 4, 5, 6], dtype);
        // Pad reaching past the spatial dims into the channel dim.
        check_reference(&[2, 3, 4, 5], &[1, 0, 0, 2, 4, 1], dtype);
    }
}

#[test]
fn test_pad_replicate_leading_dim() {
    for dtype in all_dtypes() {
        check_reference(&[2, 3, 4, 5], &[0, 0, 0, 0, 0, 0, 3, 1], dtype);
        check_reference(&[3, 8], &[0, 0, 5, 0], dtype);
    }
}

#[test]
fn test_pad_replicate_rank3_and_wide() {
    for dtype in all_dtypes() {
        check_reference(&[2, 3, 8], &[3, 5], dtype);
        check_reference(&[3, 8], &[0, 5], dtype);
        check_reference(&[40], &[16, 16], dtype);
    }
}

#[test]
fn test_pad_replicate_all_zero_padding() {
    for dtype in all_dtypes() {
        check_reference(&[3, 8], &[0, 0], dtype);
        check_reference(&[3, 8], &[0, 0, 0, 0], dtype);
        check_reference(&[2, 3, 4, 5], &[0, 0, 0, 0], dtype);
    }
}

#[test]
fn test_pad_replicate_rejects_empty_padded_dim() {
    let (client, device) = create_cpu_client();
    let tensor = Tensor::<CpuRuntime>::from_slice(&[] as &[f32], &[3, 0], &device).unwrap();
    let err = client
        .pad_mode(&tensor, &[1, 0], PadMode::Replicate)
        .unwrap_err()
        .to_string();
    assert!(err.contains("dim 1"), "error must name the dim: {err}");
    // Padding a non-empty dim of a tensor that has an empty dim elsewhere is valid.
    let out = client
        .pad_mode(&tensor, &[0, 0, 2, 2], PadMode::Replicate)
        .unwrap();
    assert_eq!(out.shape(), &[7, 0]);
}

#[test]
fn test_pad_replicate_rejects_odd_padding_length() {
    let (client, device) = create_cpu_client();
    let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
    assert!(client.pad_mode(&tensor, &[1], PadMode::Replicate).is_err());
}
