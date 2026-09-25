//! A row's CUDA convolution result does not depend on how many other rows
//! share the launch.
//!
//! The CUDA conv entry points each have more than one implementation, and the
//! rules that choose between them read the shape. Where a rule also reads
//! `batch`, a row can be handed a different kernel purely because other rows
//! joined the launch, and the row's bits move with it unless the two
//! implementations sum identically. Two kinds of choice exist:
//!
//! * **Kernel variants.** `conv1d` picks between the scalar kernel, the
//!   channel-blocked `conv1d_oc4` and the position-blocked `conv1d_ox`;
//!   `depthwise_conv2d` picks between its flat and column-blocked kernels.
//!   The `_ox` rules count threads over the whole launch, batch included, so
//!   the shapes below are sized from the device's SM count to sit BELOW the
//!   wave bound at one row and ABOVE it at eight — the kernel genuinely flips
//!   between the two runs. This is safe only because the variants form one
//!   output element's sum in the same order; the inline tests beside those
//!   launchers pin that directly.
//! * **Formulations.** conv1d, conv2d and conv_transpose1d each route
//!   well-shaped problems through a GEMM, which does NOT sum a row the way
//!   the direct kernel does. Those gates are therefore counted per batch row,
//!   and the shapes below cover the direct, im2col/GEMM, pointwise-GEMM,
//!   GEMM-first and gather-first paths.
//!
//! Run with:
//!   cd numr && cargo test --features cuda,f16 --test cuda_conv_batch_invariance

#![cfg(feature = "cuda")]

mod common;

use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode, TypeConversionOps};
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Rows in the batched launch. Eight is the decode batch the AudioVAE
/// decoder is about to use, and it clears every wave bound the shapes here
/// are sized to sit under at one row.
const BATCH: usize = 8;

/// Dtypes with a CUDA conv kernel at each accumulator width.
fn dtypes() -> Vec<DType> {
    let mut v = vec![DType::F32];
    if cfg!(feature = "f16") {
        v.push(DType::F16);
        v.push(DType::BF16);
    }
    v
}

/// Deterministic values in a range that keeps sums well away from the
/// half-precision overflow point.
fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 101) % 193) as f32) * 0.01 - 0.96)
        .collect()
}

/// A tensor of `data` in `dtype`, cast from F32 elementwise so a row's bytes
/// do not depend on the rows beside it.
fn tensor(
    client: &CudaClient,
    device: &CudaDevice,
    data: &[f32],
    shape: &[usize],
    dtype: DType,
) -> Tensor<CudaRuntime> {
    let t = Tensor::<CudaRuntime>::from_slice(data, shape, device).expect("tensor");
    if dtype == DType::F32 {
        t
    } else {
        client.cast(&t, dtype).expect("cast")
    }
}

/// Raw bits of every element, so equality is exact at any width.
fn bits(t: &Tensor<CudaRuntime>) -> Vec<u64> {
    match t.dtype() {
        DType::F32 => t
            .to_vec::<f32>()
            .iter()
            .map(|v| v.to_bits() as u64)
            .collect(),
        #[cfg(feature = "f16")]
        DType::F16 => t
            .to_vec::<half::f16>()
            .iter()
            .map(|v| v.to_bits() as u64)
            .collect(),
        #[cfg(feature = "f16")]
        DType::BF16 => t
            .to_vec::<half::bf16>()
            .iter()
            .map(|v| v.to_bits() as u64)
            .collect(),
        other => panic!("conv batch invariance: unhandled dtype {other:?}"),
    }
}

/// Every row of `batched` must be the bits its own one-row launch produced.
fn check_rows(what: &str, dtype: DType, batched: &[u64], alone: &[u64]) {
    assert_eq!(
        batched.len(),
        alone.len(),
        "{what} {dtype:?}: batched output has {} elements, the one-row runs {}",
        batched.len(),
        alone.len()
    );
    let row_len = alone.len() / BATCH;
    for (i, (b, a)) in batched.iter().zip(alone).enumerate() {
        assert_eq!(
            b,
            a,
            "{what} {dtype:?}: row {} element {} is {b:#018x} at batch {BATCH}, \
             alone it is {a:#018x}",
            i / row_len,
            i % row_len
        );
    }
}

/// One batched case: the `[BATCH, ...]` F32 data, its shape, and one row's
/// length in it.
struct Case<'a> {
    what: &'a str,
    dtype: DType,
    input: &'a [f32],
    in_shape: &'a [usize],
    row_elements: usize,
}

/// Runs `op` on the whole batch and on each row alone, and compares.
fn compare_rows<F>(case: Case<'_>, client: &CudaClient, device: &CudaDevice, op: F)
where
    F: Fn(&Tensor<CudaRuntime>) -> Tensor<CudaRuntime>,
{
    let Case {
        what,
        dtype,
        input,
        in_shape,
        row_elements,
    } = case;
    let batched = {
        let x = tensor(client, device, input, in_shape, dtype);
        bits(&op(&x))
    };

    let mut row_shape = in_shape.to_vec();
    row_shape[0] = 1;
    let mut alone: Vec<u64> = Vec::with_capacity(batched.len());
    for r in 0..BATCH {
        let row = &input[r * row_elements..(r + 1) * row_elements];
        let x = tensor(client, device, row, &row_shape, dtype);
        alone.extend(bits(&op(&x)));
    }

    check_rows(what, dtype, &batched, &alone);
}

/// SM count, or `None` when the profile is unknown — the wave bounds are
/// then trivially satisfied and no shape can straddle them.
fn compute_units(device: &CudaDevice) -> Option<usize> {
    match device.profile().compute_units {
        0 => None,
        n => Some(n as usize),
    }
}

/// A per-row thread count that lands between one eighth of the wave bound and
/// the bound itself, so one row is below it and [`BATCH`] rows are above.
fn straddling_row_threads(compute_units: usize) -> usize {
    // One wave is `compute_units * CONV_BLOCK_THREADS`; the bound is two
    // waves. A quarter of the bound leaves a factor of four of headroom on
    // each side of it.
    (compute_units * 128 * 2 / 4).max(1)
}

// ============================================================================
// conv1d
// ============================================================================

/// One conv1d case over every dtype.
#[allow(clippy::too_many_arguments)]
fn conv1d_case(
    what: &str,
    client: &CudaClient,
    device: &CudaDevice,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    length: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) {
    let row_elements = c_in * length;
    let input = values(BATCH * row_elements, 1);
    let w_data = values(c_out * (c_in / groups) * kernel, 2);
    let b_data = values(c_out, 3);

    for dtype in dtypes() {
        let weight = tensor(
            client,
            device,
            &w_data,
            &[c_out, c_in / groups, kernel],
            dtype,
        );
        let bias = tensor(client, device, &b_data, &[c_out], dtype);
        compare_rows(
            Case {
                what,
                dtype,
                input: &input,
                in_shape: &[BATCH, c_in, length],
                row_elements,
            },
            client,
            device,
            |x| {
                client
                    .conv1d(
                        x,
                        &weight,
                        Some(&bias),
                        stride,
                        PaddingMode::Valid,
                        dilation,
                        groups,
                    )
                    .expect("conv1d")
            },
        );
    }
}

#[test]
fn conv1d_rows_do_not_depend_on_the_batch() {
    common::backend_lock::with_cuda_backend(|client, device| {
        // Depthwise and narrow-group shapes sized so the position-blocked
        // kernel's wave bound falls between one row and BATCH rows. Channels
        // are fixed and the row length carries the straddle.
        if let Some(cu) = compute_units(&device) {
            let channels = 8usize;
            let kernel = 3usize;
            let blocks = (straddling_row_threads(cu) / channels).max(2);
            let output_length = blocks * 4;
            let length = output_length + kernel - 1;

            conv1d_case(
                "conv1d depthwise",
                &client,
                &device,
                channels,
                channels,
                kernel,
                length,
                1,
                1,
                channels,
            );
            // Two output channels per group: still below the oc4 floor, so
            // the same scalar-vs-position-blocked choice applies.
            conv1d_case(
                "conv1d grouped",
                &client,
                &device,
                channels,
                channels,
                kernel,
                length,
                1,
                1,
                channels / 2,
            );
        }

        // Dense, contraction below the im2col floor: the channel-blocked
        // direct kernel.
        conv1d_case(
            "conv1d dense direct",
            &client,
            &device,
            4,
            8,
            3,
            32,
            1,
            1,
            1,
        );
        // Dense, contraction 2048: the im2col + GEMM path.
        conv1d_case(
            "conv1d dense gemm",
            &client,
            &device,
            256,
            8,
            8,
            64,
            1,
            1,
            1,
        );
        // One tap, unit stride, no padding: the pointwise GEMM path.
        conv1d_case("conv1d pointwise", &client, &device, 16, 8, 1, 32, 1, 1, 1);
        // Strided and dilated, so the position-blocked kernel takes its
        // general path rather than the sliding-window one.
        conv1d_case("conv1d strided", &client, &device, 6, 6, 3, 64, 2, 2, 6);
    });
}

// ============================================================================
// conv_transpose1d
// ============================================================================

/// One conv_transpose1d case over every dtype.
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_case(
    what: &str,
    client: &CudaClient,
    device: &CudaDevice,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    length: usize,
    stride: usize,
) {
    let row_elements = c_in * length;
    let input = values(BATCH * row_elements, 4);
    // Weight is [c_in, c_out / groups, kernel] for this op; groups is 1 here,
    // which is what both GEMM formulations require.
    let w_data = values(c_in * c_out * kernel, 5);
    let b_data = values(c_out, 6);

    for dtype in dtypes() {
        let weight = tensor(client, device, &w_data, &[c_in, c_out, kernel], dtype);
        let bias = tensor(client, device, &b_data, &[c_out], dtype);
        compare_rows(
            Case {
                what,
                dtype,
                input: &input,
                in_shape: &[BATCH, c_in, length],
                row_elements,
            },
            client,
            device,
            |x| {
                client
                    .conv_transpose1d(x, &weight, Some(&bias), stride, PaddingMode::Valid, 0, 1, 1)
                    .expect("conv_transpose1d")
            },
        );
    }
}

#[test]
fn conv_transpose1d_rows_do_not_depend_on_the_batch() {
    common::backend_lock::with_cuda_backend(|client, device| {
        // c_in below the GEMM-first floor: the direct gather kernel.
        conv_transpose1d_case("conv_transpose1d direct", &client, &device, 8, 8, 4, 16, 2);
        // c_in at the GEMM-first floor, contraction far below the
        // gather-first floor: the GEMM-first path, which is what an
        // upsampling decoder takes.
        conv_transpose1d_case(
            "conv_transpose1d gemm first",
            &client,
            &device,
            32,
            16,
            4,
            64,
            2,
        );
        // Contraction 8192 and a wide output channel count, so the
        // gather-first column buffer is the smaller of the two.
        conv_transpose1d_case(
            "conv_transpose1d gather first",
            &client,
            &device,
            256,
            384,
            32,
            64,
            1,
        );
    });
}

// ============================================================================
// conv2d and depthwise_conv2d
// ============================================================================

/// One depthwise_conv2d case over every dtype.
#[allow(clippy::too_many_arguments)]
fn depthwise_conv2d_case(
    what: &str,
    client: &CudaClient,
    device: &CudaDevice,
    channels: usize,
    kernel: usize,
    height: usize,
    width: usize,
    stride: usize,
    dilation: usize,
) {
    let row_elements = channels * height * width;
    let input = values(BATCH * row_elements, 7);
    let w_data = values(channels * kernel * kernel, 8);
    let b_data = values(channels, 9);

    for dtype in dtypes() {
        let weight = tensor(
            client,
            device,
            &w_data,
            &[channels, 1, kernel, kernel],
            dtype,
        );
        let bias = tensor(client, device, &b_data, &[channels], dtype);
        compare_rows(
            Case {
                what,
                dtype,
                input: &input,
                in_shape: &[BATCH, channels, height, width],
                row_elements,
            },
            client,
            device,
            |x| {
                client
                    .depthwise_conv2d(
                        x,
                        &weight,
                        Some(&bias),
                        (stride, stride),
                        PaddingMode::Valid,
                        (dilation, dilation),
                    )
                    .expect("depthwise_conv2d")
            },
        );
    }
}

/// One dense conv2d case over every dtype.
#[allow(clippy::too_many_arguments)]
fn conv2d_case(
    what: &str,
    client: &CudaClient,
    device: &CudaDevice,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    height: usize,
    width: usize,
) {
    let row_elements = c_in * height * width;
    let input = values(BATCH * row_elements, 10);
    let w_data = values(c_out * c_in * kernel * kernel, 11);
    let b_data = values(c_out, 12);

    for dtype in dtypes() {
        let weight = tensor(
            client,
            device,
            &w_data,
            &[c_out, c_in, kernel, kernel],
            dtype,
        );
        let bias = tensor(client, device, &b_data, &[c_out], dtype);
        compare_rows(
            Case {
                what,
                dtype,
                input: &input,
                in_shape: &[BATCH, c_in, height, width],
                row_elements,
            },
            client,
            device,
            |x| {
                client
                    .conv2d(
                        x,
                        &weight,
                        Some(&bias),
                        (1, 1),
                        PaddingMode::Valid,
                        (1, 1),
                        1,
                    )
                    .expect("conv2d")
            },
        );
    }
}

#[test]
fn conv2d_rows_do_not_depend_on_the_batch() {
    common::backend_lock::with_cuda_backend(|client, device| {
        // Sized so the column-blocked depthwise kernel's wave bound falls
        // between one row and BATCH rows: channels and the output width are
        // fixed and the output height carries the straddle.
        if let Some(cu) = compute_units(&device) {
            let channels = 4usize;
            let kernel = 3usize;
            let width_blocks = 4usize; // output_w of 16, over the width floor
            let output_w = width_blocks * 4;
            let output_h = (straddling_row_threads(cu) / (channels * width_blocks)).max(2);

            depthwise_conv2d_case(
                "depthwise_conv2d",
                &client,
                &device,
                channels,
                kernel,
                output_h + kernel - 1,
                output_w + kernel - 1,
                1,
                1,
            );
            // Strided and dilated, so the blocked kernel takes its general
            // path rather than the sliding-window one.
            depthwise_conv2d_case(
                "depthwise_conv2d strided",
                &client,
                &device,
                channels,
                kernel,
                2 * output_h + 2 * kernel,
                2 * output_w + 2 * kernel,
                2,
                2,
            );
        }

        // Contraction below the conv2d im2col floor: the direct kernel.
        conv2d_case("conv2d direct", &client, &device, 2, 8, 2, 16, 16);
        // Contraction above it: the im2col + GEMM path, which chunks the
        // batch.
        conv2d_case("conv2d gemm", &client, &device, 8, 16, 3, 24, 24);
    });
}
