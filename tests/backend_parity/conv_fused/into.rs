// `conv2d_fused_into`: RRDB dense block, disjoint residual, overlap rejection.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{
    BinaryOps, Conv2dConfig, ConvActivation, ConvLayout, ConvResidual, FusedConvOps, PaddingMode,
    TypeConversionOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::{C_IN, C_OUT, KERNEL, LEAKY_SLOPE, RESIDUAL_SCALE, pattern};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, assert_tensor_allclose_tol, create_cpu_client,
    gemm_long_k_tolerance, is_dtype_supported, parity_dtypes,
};

const DENSE_BASE: usize = 4;
const DENSE_GROWTH: usize = 2;
const DENSE_CONVS: usize = 3;
const DENSE_CHANNELS: usize = DENSE_BASE + DENSE_GROWTH * DENSE_CONVS;
const DENSE_H: usize = 5;
const DENSE_W: usize = 6;
const SENTINEL: f64 = 7.0;

/// Channel dimension of a 4D tensor in `layout`.
fn channel_dim(layout: ConvLayout) -> isize {
    match layout {
        ConvLayout::Nchw => 1,
        ConvLayout::Nhwc => -1,
    }
}

fn shape_in(layout: ConvLayout, n: usize, c: usize, h: usize, w: usize) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => [n, c, h, w],
        ConvLayout::Nhwc => [n, h, w, c],
    }
}

/// Weight of `c_out x c_in` 3x3 in `layout` (OIHW or OHWI).
fn weight_in(layout: ConvLayout, c_out: usize, c_in: usize) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => [c_out, c_in, KERNEL, KERNEL],
        ConvLayout::Nhwc => [c_out, KERNEL, KERNEL, c_in],
    }
}

/// Dense buffer: channels `[0, DENSE_BASE)` hold data, the rest hold `SENTINEL`.
fn dense_data(layout: ConvLayout) -> Vec<f64> {
    let data = pattern(DENSE_BASE * DENSE_H * DENSE_W, 11);
    let pixels = DENSE_H * DENSE_W;
    (0..DENSE_CHANNELS * pixels)
        .map(|flat| {
            let (c, p) = match layout {
                ConvLayout::Nchw => (flat / pixels, flat % pixels),
                ConvLayout::Nhwc => (flat % DENSE_CHANNELS, flat / DENSE_CHANNELS),
            };
            if c < DENSE_BASE {
                data[p * DENSE_BASE + c]
            } else {
                SENTINEL
            }
        })
        .collect()
}

fn assert_exact<R: Runtime<DType = DType>>(actual: &Tensor<R>, expected: &Tensor<R>, msg: &str) {
    assert_tensor_allclose_tol(
        &actual.contiguous().unwrap(),
        &expected.contiguous().unwrap(),
        0.0,
        0.0,
        msg,
    );
}

/// Run the dense block on one backend and return the final buffer and the
/// residual conv's output, both contiguous.
///
/// Each step checks the written slice against `conv2d_fused` on a contiguous
/// copy of the input, and checks that every other channel kept the value it
/// had before the write.
fn dense_block<R, C>(
    client: &C,
    device: &R::Device,
    dtype: DType,
    layout: ConvLayout,
) -> (Tensor<R>, Tensor<R>)
where
    R: Runtime<DType = DType>,
    C: FusedConvOps<R> + BinaryOps<R> + TypeConversionOps<R>,
{
    let make = |data: &[f64], shape: &[usize]| {
        tensor_from_f64(data, shape, dtype, device, client).unwrap()
    };
    let dim = channel_dim(layout);
    let buf_shape = shape_in(layout, 1, DENSE_CHANNELS, DENSE_H, DENSE_W);
    let buf = make(&dense_data(layout), &buf_shape);
    let config = Conv2dConfig::new(layout).with_padding(PaddingMode::Same);
    let label = format!("{layout:?} [{dtype:?}]");

    for k in 0..DENSE_CONVS {
        let c_in = DENSE_BASE + DENSE_GROWTH * k;
        let w_shape = weight_in(layout, DENSE_GROWTH, c_in);
        let w = make(&pattern(w_shape.iter().product(), 20 + k), &w_shape);
        let bias = make(&pattern(DENSE_GROWTH, 30 + k), &[DENSE_GROWTH]);
        let act = ConvActivation::LeakyRelu(LEAKY_SLOPE);

        let input = buf.narrow(dim, 0, c_in).unwrap();
        let expected = client
            .conv2d_fused(
                &input.contiguous().unwrap(),
                &w,
                Some(&bias),
                &config,
                act,
                None,
            )
            .unwrap();
        let out = buf.narrow(dim, c_in, DENSE_GROWTH).unwrap();
        // A copy of the buffer, in its own storage, taken before the write.
        let before = Tensor::<R>::zeros(&buf_shape, dtype, device).unwrap();
        client.copy_into(&before, &buf).unwrap();
        client
            .conv2d_fused_into(&out, &input, &w, Some(&bias), &config, act, None)
            .unwrap_or_else(|e| panic!("dense conv {k} {label}: {e}"));

        assert_tensor_allclose(
            &out.contiguous().unwrap(),
            &expected,
            dtype,
            &format!("dense conv {k} written slice {label}"),
        );
        assert_exact(
            &buf.narrow(dim, 0, c_in).unwrap(),
            &before.narrow(dim, 0, c_in).unwrap(),
            &format!("dense conv {k} input prefix unchanged {label}"),
        );
        let tail = c_in + DENSE_GROWTH;
        if tail < DENSE_CHANNELS {
            assert_exact(
                &buf.narrow(dim, tail, DENSE_CHANNELS - tail).unwrap(),
                &before.narrow(dim, tail, DENSE_CHANNELS - tail).unwrap(),
                &format!("dense conv {k} later channels unchanged {label}"),
            );
        }
    }

    // Last conv: read every channel, write DENSE_BASE channels into a separate
    // buffer, no activation, residual = channels [0, DENSE_BASE) of the dense
    // buffer: out = x + 0.2 * conv.
    let w_shape = weight_in(layout, DENSE_BASE, DENSE_CHANNELS);
    let w = make(&pattern(w_shape.iter().product(), 40), &w_shape);
    let bias = make(&pattern(DENSE_BASE, 41), &[DENSE_BASE]);
    let residual = buf.narrow(dim, 0, DENSE_BASE).unwrap();
    let residual_copy = residual.contiguous().unwrap();
    let out_shape = shape_in(layout, 1, DENSE_BASE, DENSE_H, DENSE_W);
    let expected = client
        .conv2d_fused(
            &buf.contiguous().unwrap(),
            &w,
            Some(&bias),
            &config,
            ConvActivation::None,
            Some(ConvResidual {
                tensor: &residual_copy,
                scale: RESIDUAL_SCALE,
            }),
        )
        .unwrap();
    let out = Tensor::<R>::zeros(&out_shape, dtype, device).unwrap();
    client
        .conv2d_fused_into(
            &out,
            &buf,
            &w,
            Some(&bias),
            &config,
            ConvActivation::None,
            Some(ConvResidual {
                tensor: &residual,
                scale: RESIDUAL_SCALE,
            }),
        )
        .unwrap_or_else(|e| panic!("dense residual conv {label}: {e}"));
    assert_tensor_allclose(
        &out,
        &expected,
        dtype,
        &format!("dense residual conv {label}"),
    );
    (buf.contiguous().unwrap(), out)
}

#[test]
fn test_conv2d_fused_into_dense_block() {
    let (client, device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        for layout in [ConvLayout::Nhwc, ConvLayout::Nchw] {
            let (cpu_buf, cpu_out) = dense_block(&client, &device, dtype, layout);
            let k = DENSE_CHANNELS * KERNEL * KERNEL;
            let (rtol, atol) = gemm_long_k_tolerance(dtype, k, 1.0);

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|client, device| {
                    let (buf, out) = dense_block(&client, &device, dtype, layout);
                    let msg = format!("dense block {layout:?} CUDA vs CPU [{dtype:?}]");
                    assert_tensor_allclose_tol(&buf, &cpu_buf, rtol, atol, &msg);
                    assert_tensor_allclose_tol(&out, &cpu_out, rtol, atol, &msg);
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|client, device| {
                    let (buf, out) = dense_block(&client, &device, dtype, layout);
                    let msg = format!("dense block {layout:?} WebGPU vs CPU [{dtype:?}]");
                    assert_tensor_allclose_tol(&buf, &cpu_buf, rtol, atol, &msg);
                    assert_tensor_allclose_tol(&out, &cpu_out, rtol, atol, &msg);
                });
            }
        }
    }
}

/// Residual and `out` are disjoint channel slices of one buffer.
fn residual_beside_out<R, C>(client: &C, device: &R::Device, dtype: DType) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: FusedConvOps<R> + TypeConversionOps<R>,
{
    let make = |data: &[f64], shape: &[usize]| {
        tensor_from_f64(data, shape, dtype, device, client).unwrap()
    };
    let buf_shape = [1, DENSE_H, DENSE_W, 2 * C_OUT];
    let buf_data = pattern(buf_shape.iter().product(), 50);
    let buf = make(&buf_data, &buf_shape);
    let pristine = make(&buf_data, &buf_shape);
    let x = make(
        &pattern(DENSE_H * DENSE_W * C_IN, 51),
        &[1, DENSE_H, DENSE_W, C_IN],
    );
    let w_shape = weight_in(ConvLayout::Nhwc, C_OUT, C_IN);
    let w = make(&pattern(w_shape.iter().product(), 52), &w_shape);
    let config = Conv2dConfig::new(ConvLayout::Nhwc).with_padding(PaddingMode::Same);

    let residual = buf.narrow(-1, 0, C_OUT).unwrap();
    let out = buf.narrow(-1, C_OUT, C_OUT).unwrap();
    let expected = client
        .conv2d_fused(
            &x,
            &w,
            None,
            &config,
            ConvActivation::Relu,
            Some(ConvResidual {
                tensor: &residual.contiguous().unwrap(),
                scale: RESIDUAL_SCALE,
            }),
        )
        .unwrap();
    client
        .conv2d_fused_into(
            &out,
            &x,
            &w,
            None,
            &config,
            ConvActivation::Relu,
            Some(ConvResidual {
                tensor: &residual,
                scale: RESIDUAL_SCALE,
            }),
        )
        .unwrap_or_else(|e| panic!("residual beside out [{dtype:?}]: {e}"));
    assert_tensor_allclose(
        &out.contiguous().unwrap(),
        &expected,
        dtype,
        &format!("residual beside out [{dtype:?}]"),
    );
    assert_exact(
        &residual,
        &pristine.narrow(-1, 0, C_OUT).unwrap(),
        &format!("residual slice unchanged [{dtype:?}]"),
    );
    buf.contiguous().unwrap()
}

#[test]
fn test_conv2d_fused_into_residual_in_same_buffer() {
    let (client, device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let cpu_buf = residual_beside_out(&client, &device, dtype);
        let (rtol, atol) = gemm_long_k_tolerance(dtype, C_IN * KERNEL * KERNEL, 1.0);

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let buf = residual_beside_out(&client, &device, dtype);
                let msg = format!("residual beside out CUDA vs CPU [{dtype:?}]");
                assert_tensor_allclose_tol(&buf, &cpu_buf, rtol, atol, &msg);
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let buf = residual_beside_out(&client, &device, dtype);
                let msg = format!("residual beside out WebGPU vs CPU [{dtype:?}]");
                assert_tensor_allclose_tol(&buf, &cpu_buf, rtol, atol, &msg);
            });
        }
    }
}

// ============================================================================
// Overlap rejection
// ============================================================================

fn assert_overlap_error(result: numr::error::Result<()>, operand: &str) {
    match result {
        Err(Error::InvalidArgument { arg, reason }) => {
            assert_eq!(arg, "out");
            assert!(
                reason.contains(operand),
                "reason must name {operand}: {reason}"
            );
        }
        other => panic!("expected overlap error naming {operand}, got {other:?}"),
    }
}

#[test]
fn test_conv2d_fused_into_rejects_out_overlapping_input() {
    let (client, device) = create_cpu_client();
    let dtype = DType::F32;
    let make = |data: &[f64], shape: &[usize]| {
        tensor_from_f64(data, shape, dtype, &device, &client).unwrap()
    };
    let buf_data = pattern(DENSE_H * DENSE_W * 6, 60);
    let buf = make(&buf_data, &[1, DENSE_H, DENSE_W, 6]);
    let pristine = make(&buf_data, &[1, DENSE_H, DENSE_W, 6]);
    let w = make(&pattern(2 * 9 * 4, 61), &[2, KERNEL, KERNEL, 4]);
    let config = Conv2dConfig::new(ConvLayout::Nhwc).with_padding(PaddingMode::Same);

    let input = buf.narrow(-1, 0, 4).unwrap();
    let out = buf.narrow(-1, 3, 2).unwrap();
    let act = ConvActivation::None;
    let result = client.conv2d_fused_into(&out, &input, &w, None, &config, act, None);
    assert_overlap_error(result, "input");
    assert_exact(&buf, &pristine, "rejected call leaves the buffer unchanged");
}

#[test]
fn test_conv2d_fused_into_rejects_out_overlapping_residual() {
    let (client, device) = create_cpu_client();
    let dtype = DType::F32;
    let make = |data: &[f64], shape: &[usize]| {
        tensor_from_f64(data, shape, dtype, &device, &client).unwrap()
    };
    let buf = make(
        &pattern(DENSE_H * DENSE_W * 4, 70),
        &[1, DENSE_H, DENSE_W, 4],
    );
    let x = make(
        &pattern(DENSE_H * DENSE_W * 4, 71),
        &[1, DENSE_H, DENSE_W, 4],
    );
    let w = make(&pattern(2 * 9 * 4, 72), &[2, KERNEL, KERNEL, 4]);
    let config = Conv2dConfig::new(ConvLayout::Nhwc).with_padding(PaddingMode::Same);

    let out = buf.narrow(-1, 0, 2).unwrap();
    let residual = buf.narrow(-1, 1, 2).unwrap();
    let result = client.conv2d_fused_into(
        &out,
        &x,
        &w,
        None,
        &config,
        ConvActivation::None,
        Some(ConvResidual {
            tensor: &residual,
            scale: RESIDUAL_SCALE,
        }),
    );
    assert_overlap_error(result, "residual");
}
