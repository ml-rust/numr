// Backend parity for `BinaryOps::copy_into` with strided source and strided
// destination views.
//
// The op is pure data movement, so every backend must match the CPU result
// EXACTLY (`rtol = atol = 0.0`). The CPU result is itself checked against a
// plain-Rust model of the logical-index mapping, so a bug shared by all
// backends cannot pass. The model also proves that elements outside the
// destination view keep their original values.
//
// Dtypes run under `DTypeDomain::AllNumeric`. CUDA covers every width. WebGPU
// covers its 32-bit scope (F32, I32, U32). Values stay below 97 so they fit
// every dtype, including I8.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{BinaryOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose_tol, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Values that vary with the flat index and stay below 97.
fn pattern(n: usize, salt: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 37 + salt) % 97) as f64).collect()
}

/// One destination/source view pair built from a buffer and a source base.
#[derive(Clone, Copy, Debug)]
enum Scenario {
    /// `buf.narrow(-1, 2, 3)` of a `[2, 3, 4, 8]` buffer, contiguous source.
    ChannelSlice,
    /// Destination `buf.permute([2, 0, 1])` of a `[4, 5, 6]` buffer.
    PermutedDestination,
    /// Permuted source into a channel slice of a larger buffer.
    StridedSourceAndDestination,
    /// Destination flipped along its last dimension (negative stride).
    FlippedDestination,
}

const SCENARIOS: [Scenario; 4] = [
    Scenario::ChannelSlice,
    Scenario::PermutedDestination,
    Scenario::StridedSourceAndDestination,
    Scenario::FlippedDestination,
];

impl Scenario {
    fn buffer_shape(self) -> &'static [usize] {
        match self {
            Self::ChannelSlice => &[2, 3, 4, 8],
            Self::PermutedDestination => &[4, 5, 6],
            Self::StridedSourceAndDestination => &[4, 3, 8],
            Self::FlippedDestination => &[3, 7],
        }
    }

    fn source_shape(self) -> &'static [usize] {
        match self {
            Self::ChannelSlice => &[2, 3, 4, 3],
            Self::PermutedDestination => &[6, 4, 5],
            Self::StridedSourceAndDestination => &[5, 4, 3],
            Self::FlippedDestination => &[3, 7],
        }
    }

    fn views<R: Runtime<DType = DType>>(
        self,
        buf: &Tensor<R>,
        src_base: &Tensor<R>,
    ) -> (Tensor<R>, Tensor<R>) {
        match self {
            Self::ChannelSlice => (buf.narrow(-1, 2, 3).unwrap(), src_base.clone()),
            Self::PermutedDestination => (buf.permute(&[2, 0, 1]).unwrap(), src_base.clone()),
            Self::StridedSourceAndDestination => (
                buf.narrow(-1, 1, 5).unwrap(),
                src_base.permute(&[1, 2, 0]).unwrap(),
            ),
            Self::FlippedDestination => (buf.flip(-1).unwrap(), src_base.clone()),
        }
    }
}

/// Plain-Rust model of `copy_into`: writes each logical element of the source
/// to the matching logical position of the destination.
fn model<R: Runtime<DType = DType>>(
    buf_data: &[f64],
    src_data: &[f64],
    dst: &Tensor<R>,
    src: &Tensor<R>,
) -> Vec<f64> {
    let mut out = buf_data.to_vec();
    let shape = dst.shape();
    let numel: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        let at = |t: &Tensor<R>| {
            let off: isize = t.offset() as isize
                + index
                    .iter()
                    .zip(t.strides())
                    .map(|(&i, &s)| i as isize * s)
                    .sum::<isize>();
            off as usize
        };
        out[at(dst)] = src_data[at(src)];
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    out
}

/// Run one scenario on a backend and return the whole buffer afterwards.
fn run<R, C>(scenario: Scenario, dtype: DType, device: &R::Device, client: &C) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + TypeConversionOps<R>,
{
    let buf_data = pattern(scenario.buffer_shape().iter().product(), 5);
    let src_data = pattern(scenario.source_shape().iter().product(), 11);
    let buf = tensor_from_f64(&buf_data, scenario.buffer_shape(), dtype, device, client)
        .unwrap_or_else(|e| panic!("buffer failed for {scenario:?} [{dtype:?}]: {e}"));
    let src_base = tensor_from_f64(&src_data, scenario.source_shape(), dtype, device, client)
        .unwrap_or_else(|e| panic!("source failed for {scenario:?} [{dtype:?}]: {e}"));
    let (dst, src) = scenario.views(&buf, &src_base);
    let before = buf.ptr();
    client
        .copy_into(&dst, &src)
        .unwrap_or_else(|e| panic!("copy_into failed for {scenario:?} [{dtype:?}]: {e}"));
    assert_eq!(buf.ptr(), before, "{scenario:?} [{dtype:?}]: buffer moved");
    buf
}

fn assert_scenario_parity(scenario: Scenario) {
    let buf_data = pattern(scenario.buffer_shape().iter().product(), 5);
    let src_data = pattern(scenario.source_shape().iter().product(), 11);

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_result = run(scenario, dtype, &cpu_device, &cpu_client);

        // CPU against the model: written channels and untouched channels.
        let cpu_buf = tensor_from_f64(
            &buf_data,
            scenario.buffer_shape(),
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap();
        let cpu_src = tensor_from_f64(
            &src_data,
            scenario.source_shape(),
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap();
        let (dst, src) = scenario.views(&cpu_buf, &cpu_src);
        let expected = tensor_from_f64(
            &model(&buf_data, &src_data, &dst, &src),
            scenario.buffer_shape(),
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap();
        assert_tensor_allclose_tol(
            &cpu_result,
            &expected,
            0.0,
            0.0,
            &format!("{scenario:?} CPU vs model [{dtype:?}]"),
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let result = run(scenario, dtype, &device, &client);
                assert_tensor_allclose_tol(
                    &result,
                    &expected,
                    0.0,
                    0.0,
                    &format!("{scenario:?} CUDA vs model [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let result = run(scenario, dtype, &device, &client);
                assert_tensor_allclose_tol(
                    &result,
                    &expected,
                    0.0,
                    0.0,
                    &format!("{scenario:?} WebGPU vs model [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn copy_into_channel_slice_parity() {
    assert_scenario_parity(SCENARIOS[0]);
}

#[test]
fn copy_into_permuted_destination_parity() {
    assert_scenario_parity(SCENARIOS[1]);
}

#[test]
fn copy_into_strided_source_and_destination_parity() {
    assert_scenario_parity(SCENARIOS[2]);
}

#[test]
fn copy_into_flipped_destination_parity() {
    assert_scenario_parity(SCENARIOS[3]);
}

/// Overlap rejection, identical-view no-op and zero-size no-op on one backend.
fn check_edge_cases<R, C>(dtype: DType, device: &R::Device, client: &C)
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + TypeConversionOps<R>,
{
    let data = pattern(2 * 3 * 4 * 8, 5);
    let buf = tensor_from_f64::<R>(&data, &[2, 3, 4, 8], dtype, device, client).unwrap();

    // Spans [0, ..] and [3, ..] of one buffer intersect, so the call is rejected.
    let out = buf.narrow(-1, 0, 3).unwrap();
    let src = buf.narrow(-1, 3, 3).unwrap();
    assert!(
        matches!(
            client.copy_into(&out, &src),
            Err(Error::InvalidArgument { arg: "out", .. })
        ),
        "overlap not rejected [{dtype:?}]"
    );

    // A contiguous destination is checked too: rows 0..2 and 1..3 share row 1.
    let rows = tensor_from_f64::<R>(&pattern(24, 5), &[4, 6], dtype, device, client).unwrap();
    let rows_out = rows.narrow(0, 0, 2).unwrap();
    let rows_src = rows.narrow(0, 1, 2).unwrap();
    assert!(
        matches!(
            client.copy_into(&rows_out, &rows_src),
            Err(Error::InvalidArgument { arg: "out", .. })
        ),
        "contiguous overlap not rejected [{dtype:?}]"
    );
    // Disjoint rows of one buffer copy fine.
    let rows_tail = rows.narrow(0, 2, 2).unwrap();
    client
        .copy_into(&rows_out, &rows_tail)
        .unwrap_or_else(|e| panic!("disjoint rows rejected [{dtype:?}]: {e}"));

    // Identical strided view: no-op, buffer keeps its values.
    let data = pattern(2 * 3 * 4 * 8, 5);
    let fresh = tensor_from_f64::<R>(&data, &[2, 3, 4, 8], dtype, device, client).unwrap();
    let view = fresh.narrow(-1, 2, 3).unwrap();
    client
        .copy_into(&view, &view)
        .unwrap_or_else(|e| panic!("identical view failed [{dtype:?}]: {e}"));
    let (cpu_client, cpu_device) = create_cpu_client();
    let expected = tensor_from_f64(&data, &[2, 3, 4, 8], dtype, &cpu_device, &cpu_client).unwrap();
    assert_tensor_allclose_tol(
        &fresh,
        &expected,
        0.0,
        0.0,
        &format!("identical view no-op [{dtype:?}]"),
    );

    // Zero-size tensors: no-op.
    let empty_out = Tensor::<R>::zeros(&[2, 0, 3], dtype, device).unwrap();
    let empty_src = Tensor::<R>::zeros(&[2, 0, 3], dtype, device).unwrap();
    client
        .copy_into(&empty_out, &empty_src)
        .unwrap_or_else(|e| panic!("zero-size copy failed [{dtype:?}]: {e}"));
}

#[test]
fn copy_into_edge_cases_parity() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        check_edge_cases(dtype, &cpu_device, &cpu_client);

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| check_edge_cases(dtype, &device, &client));
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| check_edge_cases(dtype, &device, &client));
        }
    }
}
