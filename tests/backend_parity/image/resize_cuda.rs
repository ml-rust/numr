#![cfg(feature = "cuda")]
// ImageOps::resize on CUDA against PyTorch CPU references, and against the
// CPU backend bit for bit.

use numr::dtype::DType;
use numr::ops::{ConvLayout, ImageOps, ResizeConfig, ResizeFilter};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use super::resize_golden::{Kind, blocks, input, shape, to_layout};
#[cfg(feature = "f16")]
use super::resize_suite::check_resize_half;
use super::resize_suite::{LAYOUTS, check_resize_f64, check_resize_golden, check_resize_srvgg};
use super::resize_suite_views::{
    check_resize_equal_sizes, check_resize_errors, check_resize_into_channel_slice,
    check_resize_layouts_agree, check_resize_strided_input,
};
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::create_cpu_client;

#[test]
fn test_resize_cuda_f32_matches_torch() {
    with_cuda_backend(|client, device| {
        check_resize_golden::<CudaRuntime, _>(&client, &device, 1.0);
        check_resize_srvgg::<CudaRuntime, _>(&client, &device, 1.0);
        check_resize_f64::<CudaRuntime, _>(&client, &device);
    });
}

#[cfg(feature = "f16")]
#[test]
fn test_resize_cuda_half_rounds_once() {
    with_cuda_backend(|client, device| {
        check_resize_half::<CudaRuntime, _>(&client, &device);
    });
}

#[test]
fn test_resize_cuda_views_equal_sizes_and_errors() {
    with_cuda_backend(|client, device| {
        check_resize_strided_input::<CudaRuntime, _>(&client, &device);
        check_resize_into_channel_slice::<CudaRuntime, _>(&client, &device);
        check_resize_layouts_agree::<CudaRuntime, _>(&client, &device);
        check_resize_equal_sizes::<CudaRuntime, _>(&client, &device);
        check_resize_errors::<CudaRuntime, _>(&client, &device);
    });
}

/// Resize `x` (NCHW values, extents `dims`) stored as `dtype` in `layout`,
/// and return the raw output bytes.
fn run_bytes<R: Runtime<DType = DType>, C: ImageOps<R>>(
    client: &C,
    device: &R::Device,
    x: &[f32],
    dims: [usize; 4],
    dtype: DType,
    cfg: &ResizeConfig,
    layout: ConvLayout,
) -> Vec<u8> {
    let laid = to_layout(x, dims, layout);
    let s = shape(dims, layout);
    let t = match dtype {
        DType::F32 => Tensor::<R>::from_slice(&laid, &s, device),
        DType::F64 => {
            let v: Vec<f64> = laid.iter().map(|&f| f64::from(f)).collect();
            Tensor::<R>::from_slice(&v, &s, device)
        }
        #[cfg(feature = "f16")]
        DType::F16 => {
            let v: Vec<half::f16> = laid.iter().map(|&f| half::f16::from_f32(f)).collect();
            Tensor::<R>::from_slice(&v, &s, device)
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            let v: Vec<half::bf16> = laid.iter().map(|&f| half::bf16::from_f32(f)).collect();
            Tensor::<R>::from_slice(&v, &s, device)
        }
        other => panic!("no input builder for {other:?}"),
    }
    .expect("input tensor");
    client
        .resize(&t, cfg, layout)
        .unwrap_or_else(|e| panic!("{dtype:?} {cfg:?} {layout:?}: {e}"))
        .to_bytes()
        .expect("readback")
}

/// CUDA and CPU give the same bits for every reference, a larger batch, and
/// every dtype CUDA resizes.
#[test]
fn test_resize_cuda_matches_cpu_bit_for_bit() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        let mut cases: Vec<([usize; 4], ResizeConfig)> = blocks()
            .iter()
            .filter(|b| b.kind == Kind::F32)
            .map(|b| (b.dims, b.config()))
            .collect();
        for size in [(61, 29), (12, 100), (37, 53)] {
            for filter in [ResizeFilter::Bilinear, ResizeFilter::Bicubic] {
                for aa in [false, true] {
                    let cfg = ResizeConfig::new(size, filter).with_antialias(aa);
                    cases.push(([2, 5, 37, 53], cfg));
                }
            }
        }
        let mut dtypes = vec![DType::F32, DType::F64];
        if cfg!(feature = "f16") {
            dtypes.extend([DType::F16, DType::BF16]);
        }
        for (dims, cfg) in &cases {
            let x = input(*dims);
            for &dtype in &dtypes {
                for layout in LAYOUTS {
                    let want = run_bytes::<CpuRuntime, _>(
                        &cpu_client,
                        &cpu_device,
                        &x,
                        *dims,
                        dtype,
                        cfg,
                        layout,
                    );
                    let got = run_bytes::<CudaRuntime, _>(
                        &cuda_client,
                        &cuda_device,
                        &x,
                        *dims,
                        dtype,
                        cfg,
                        layout,
                    );
                    if got != want {
                        let size = dtype.size_in_bytes();
                        let first = got
                            .chunks(size)
                            .zip(want.chunks(size))
                            .position(|(g, w)| g != w)
                            .unwrap_or(0);
                        panic!(
                            "{dims:?} {cfg:?} {dtype:?} {layout:?}: CUDA differs from CPU, \
                             first at element {first}"
                        );
                    }
                }
            }
        }
    });
}
