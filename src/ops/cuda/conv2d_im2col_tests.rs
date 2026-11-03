//! Unit tests of the im2col + GEMM conv2d path. They call
//! `conv2d_im2col_capped` directly, so F16 and BF16 shapes that `conv2d`
//! routes to the native implicit-GEMM kernel still cover this path.

use super::*;
use crate::ops::conv_common::validate_conv2d;
use crate::ops::{PaddingMode, RandomOps, TypeConversionOps};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaDevice;

fn setup() -> Option<CudaClient> {
    if !crate::runtime::cuda::is_cuda_available() {
        return None;
    }
    let device = CudaDevice::new(0);
    Some(CudaRuntime::default_client(&device))
}

/// `Conv2dParams` for a `groups == 1` conv2d at `dtype`, with bias.
fn params_for(
    input_shape: &[usize],
    weight_shape: &[usize],
    padding: PaddingMode,
    dtype: DType,
) -> Conv2dParams {
    validate_conv2d(
        input_shape,
        weight_shape,
        Some(&weight_shape[..1]),
        (1, 1),
        padding,
        (1, 1),
        1,
        dtype,
        dtype,
        Some(dtype),
    )
    .expect("params")
}

/// Element bits of `t`, widened to `u64`. A half tensor is cast to F32
/// first: that cast is exact and one-to-one, so equal F32 bits mean equal
/// half bits.
fn element_bits(client: &CudaClient, t: &Tensor<CudaRuntime>) -> Vec<u64> {
    match t.dtype() {
        DType::F64 => t.to_vec::<f64>().iter().map(|v| v.to_bits()).collect(),
        DType::F32 => t
            .to_vec::<f32>()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        _ => client
            .cast(t, DType::F32)
            .expect("cast to F32")
            .to_vec::<f32>()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
    }
}

/// The default budget admits the motivating shape, 64 to 64 channels at
/// 3x3 over 1080x1920, whose one-row column exceeds `MAX_COL_ELEMENTS`.
/// It splits into aligned windows of one row each.
#[test]
fn large_column_takes_gemm_in_windows() {
    let params = params_for(
        &[1, 64, 1080, 1920],
        &[64, 64, 3, 3],
        PaddingMode::Custom(1, 1, 1, 1),
        DType::F32,
    );
    let contraction = 64 * 3 * 3;
    assert!(contraction * 1080 * 1920 > MAX_COL_ELEMENTS);
    assert!(use_conv2d_im2col(&params, DType::F32));

    let plan = chunk_plan(&params, MAX_COL_ELEMENTS);
    assert_eq!(plan.rows, 1);
    assert_eq!(plan.width % WINDOW_ALIGN, 0);
    assert!(plan.width * contraction <= MAX_COL_ELEMENTS);
    assert!(plan.width < 1080 * 1920);
}

/// The gate keeps its floors and its group rule: a column far over the
/// budget does not lift them.
#[test]
fn gate_keeps_floors_at_any_column_size() {
    let shallow = params_for(
        &[1, 2, 4000, 4000],
        &[8, 2, 3, 3],
        PaddingMode::Valid,
        DType::F32,
    );
    assert!(!use_conv2d_im2col(&shallow, DType::F32));
    let narrow = params_for(
        &[1, 64, 2000, 2000],
        &[MIN_C_OUT - 1, 64, 3, 3],
        PaddingMode::Valid,
        DType::F32,
    );
    assert!(!use_conv2d_im2col(&narrow, DType::F32));
}

/// Each budget below lands on a different chunk shape for the shape the
/// bitwise test runs: 3 rows of 20x20 output positions, contraction 36.
#[test]
fn chunk_plan_shapes() {
    let params = params_for(
        &[3, 4, 20, 20],
        &[80, 4, 3, 3],
        PaddingMode::Custom(1, 1, 1, 1),
        DType::F32,
    );
    let (k, spatial) = (36usize, 400usize);
    let plan = |cap| {
        let p = chunk_plan(&params, cap);
        (p.rows, p.width)
    };
    assert_eq!(plan(usize::MAX), (3, spatial));
    assert_eq!(plan(3 * k * spatial), (3, spatial));
    assert_eq!(plan(2 * k * spatial), (2, spatial));
    assert_eq!(plan(k * 300), (1, 256));
    assert_eq!(plan(k * 70), (1, 70));
    assert_eq!(plan(k * 7), (1, 7));
    assert_eq!(plan(1), (1, 1));
}

/// A chunk boundary moves an output element to another GEMM launch, tile
/// or batch slot, never to another accumulation order (see
/// [`chunk_plan`]). Every chunked run must reproduce the one-chunk run bit
/// for bit, at every dtype the path takes. The budgets cover ragged batch
/// chunks, aligned windows with a ragged tail, and unaligned windows.
/// `c_out = 80` puts the F32 GEMM on a different tile for windows up to
/// 64 wide than for wider ones.
#[test]
fn chunked_matches_one_chunk_bitwise() {
    let Some(client) = setup() else {
        return;
    };
    let (batch, c_in, c_out, k, h, w) = (3usize, 4usize, 80usize, 3usize, 20usize, 20usize);
    let contraction = c_in * k * k;
    let spatial = h * w;
    let caps = [
        2 * contraction * spatial,
        contraction * 300,
        contraction * 70,
        contraction * 7,
    ];

    let mut dtypes = vec![DType::F32, DType::F64];
    if cfg!(feature = "f16") {
        dtypes.extend([DType::F16, DType::BF16]);
    }
    for dtype in dtypes {
        let input = client.rand(&[batch, c_in, h, w], dtype).expect("input");
        let weight = client.rand(&[c_out, c_in, k, k], dtype).expect("weight");
        let bias = client.rand(&[c_out], dtype).expect("bias");
        let params = params_for(
            input.shape(),
            weight.shape(),
            PaddingMode::Custom(1, 1, 1, 1),
            dtype,
        );

        let whole =
            conv2d_im2col_capped(&client, &input, &weight, Some(&bias), &params, usize::MAX)
                .expect("whole");
        let whole_bits = element_bits(&client, &whole);
        assert_eq!(whole_bits.len(), batch * c_out * spatial);

        for cap in caps {
            let chunked = conv2d_im2col_capped(&client, &input, &weight, Some(&bias), &params, cap)
                .expect("chunked");
            assert_eq!(whole.shape(), chunked.shape());
            let chunked_bits = element_bits(&client, &chunked);
            for (i, (x, y)) in whole_bits.iter().zip(&chunked_bits).enumerate() {
                assert_eq!(
                    x, y,
                    "{dtype:?} cap {cap}: element {i} bits {x:#x} vs {y:#x}"
                );
            }
        }
    }
}

/// Every element of `t` as F64. A half tensor widens exactly through F32.
fn host_f64(client: &CudaClient, t: &Tensor<CudaRuntime>) -> Vec<f64> {
    client
        .cast(t, DType::F32)
        .expect("cast to F32")
        .to_vec::<f32>()
        .into_iter()
        .map(f64::from)
        .collect()
}

/// F16 and BF16 im2col + GEMM against the CPU conv2d in F64 on the same
/// values. `conv2d` routes these shapes to the native kernel on sm_75+, so
/// this is the half-dtype parity check of the im2col path itself. The
/// shapes are those of `tests/backend_parity/conv2d_im2col.rs`, plus a
/// wider one.
///
/// The GEMM result rounds once to the storage dtype and the bias add
/// rounds again, so `rtol` is two storage ulps relative. The F32
/// accumulation error sits far below `atol`.
#[test]
fn half_im2col_matches_cpu_reference() {
    use crate::ops::ConvOps;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    let Some(client) = setup() else {
        return;
    };
    if !cfg!(feature = "f16") {
        return;
    }
    let cpu_device = CpuDevice::new();
    let cpu = CpuRuntime::default_client(&cpu_device);
    // (input, weight, stride, padding, dilation)
    let cases = [
        (
            [1, 16, 6, 6],
            [4, 16, 4, 4],
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
        ),
        (
            [1, 16, 10, 8],
            [4, 16, 8, 4],
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
        ),
        (
            [1, 16, 9, 11],
            [4, 16, 4, 4],
            (2, 3),
            PaddingMode::Custom(1, 0, 2, 1),
            (2, 2),
        ),
        (
            [2, 16, 6, 6],
            [4, 16, 4, 4],
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
        ),
        (
            [1, 16, 6, 10],
            [4, 16, 4, 4],
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
        ),
        (
            [2, 64, 9, 13],
            [48, 64, 3, 3],
            (1, 1),
            PaddingMode::Custom(1, 1, 1, 1),
            (1, 1),
        ),
    ];
    for dtype in [DType::F16, DType::BF16] {
        let rtol = match dtype {
            DType::F16 => 2f64.powi(-9),
            _ => 2f64.powi(-6),
        };
        for (x_shape, w_shape, stride, padding, dilation) in cases {
            let label = format!("{x_shape:?} * {w_shape:?} stride {stride:?} [{dtype:?}]");
            let x = client.rand(&x_shape, dtype).expect("input");
            let w = client.rand(&w_shape, dtype).expect("weight");
            let b = client.rand(&w_shape[..1], dtype).expect("bias");
            let params = validate_conv2d(
                &x_shape,
                &w_shape,
                Some(&w_shape[..1]),
                stride,
                padding,
                dilation,
                1,
                dtype,
                dtype,
                Some(dtype),
            )
            .expect("params");
            let got = conv2d_im2col(&client, &x, &w, Some(&b), &params).expect("im2col");
            let got = host_f64(&client, &got);

            let host = |t: &Tensor<CudaRuntime>, shape: &[usize]| {
                Tensor::<CpuRuntime>::from_slice(&host_f64(&client, t), shape, &cpu_device)
                    .expect("host tensor")
            };
            let expected = cpu
                .conv2d(
                    &host(&x, &x_shape),
                    &host(&w, &w_shape),
                    Some(&host(&b, &w_shape[..1])),
                    stride,
                    padding,
                    dilation,
                    1,
                )
                .expect("CPU conv2d")
                .to_vec::<f64>();
            assert_eq!(got.len(), expected.len(), "{label}: length");
            for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                let tol = 1e-3 + rtol * e.abs();
                assert!(
                    (g - e).abs() <= tol,
                    "{label}: element {i}: im2col {g} vs F64 reference {e} (tol {tol:.3e})"
                );
            }
        }
    }
}
