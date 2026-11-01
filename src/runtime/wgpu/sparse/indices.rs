//! Index dtype bridging and pointer scans for the WebGPU sparse kernels.
//!
//! `CsrData`, `CscData` and `CooData` hold I64 indices on every runtime. WGSL
//! has no 64-bit integers, so every kernel binds indices as `array<i32>`.
//! Each entry point casts I64 indices to I32 on the GPU before a launch and
//! casts output indices back to I64. No index data leaves the device.

use wgpu::Buffer;

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::launch_exclusive_scan_i32;
use super::super::{WgpuClient, WgpuRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Return `indices` as a contiguous I32 tensor for an `array<i32>` binding.
///
/// I64 input is cast on the GPU. I32 input passes through. Any other dtype
/// returns `UnsupportedDType` naming `op`.
pub(super) fn indices_to_i32(
    client: &WgpuClient,
    indices: &Tensor<WgpuRuntime>,
    op: &'static str,
) -> Result<Tensor<WgpuRuntime>> {
    match indices.dtype() {
        DType::I32 => indices.contiguous(),
        DType::I64 if indices.numel() == 0 => {
            Tensor::<WgpuRuntime>::zeros(&[0], DType::I32, client.device())
        }
        DType::I64 => client.cast(indices, DType::I32),
        dtype => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Return kernel output indices as the I64 tensor the sparse formats require.
pub(super) fn indices_to_i64(
    client: &WgpuClient,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    match indices.dtype() {
        DType::I64 => Ok(indices.clone()),
        _ if indices.numel() == 0 => {
            Tensor::<WgpuRuntime>::zeros(&[0], DType::I64, client.device())
        }
        _ => client.cast(indices, DType::I64),
    }
}

/// Check that `dtype` is a 32-bit value dtype that WebGPU sparse kernels
/// support: F32, I32 or U32.
pub(super) fn check_value_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::I32 | DType::U32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Return `tensor`, or a one-element zero tensor of its dtype when it is
/// empty. wgpu refuses a zero-size binding. A kernel never reads the
/// placeholder, because the pointers it reads describe no entries.
pub(super) fn bindable(
    client: &WgpuClient,
    tensor: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    if tensor.numel() == 0 {
        Tensor::<WgpuRuntime>::zeros(&[1], tensor.dtype(), client.device())
    } else {
        tensor.contiguous()
    }
}

/// Create a 16-byte uniform buffer holding `words`.
pub(super) fn uniform_words(client: &WgpuClient, label: &str, words: [u32; 4]) -> Buffer {
    let buffer = client.create_uniform_buffer(label, 16);
    client.write_buffer(&buffer, &words);
    buffer
}

/// Exclusive scan of `n` I32 counts on the GPU.
///
/// Returns the `n + 1` I32 pointers and their total, `ptrs[n]`. Reading the
/// total moves one `i32` to the host: the caller needs it to size the
/// output.
pub(super) fn scan_counts(
    client: &WgpuClient,
    counts: &Tensor<WgpuRuntime>,
    n: usize,
) -> Result<(Tensor<WgpuRuntime>, usize)> {
    let ptrs = Tensor::<WgpuRuntime>::zeros(&[n + 1], DType::I32, client.device())?;
    if n == 0 {
        return Ok((ptrs, 0));
    }
    let params = uniform_words(client, "sparse_scan_params", [n as u32, 0, 0, 0]);
    let counts_buf = get_tensor_buffer(counts)?;
    let ptrs_buf = get_tensor_buffer(&ptrs)?;
    launch_exclusive_scan_i32(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &counts_buf,
        &ptrs_buf,
        &params,
    )?;

    let staging = client.create_staging_buffer("sparse_scan_total", 4);
    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sparse_scan_total"),
            });
    encoder.copy_buffer_to_buffer(&ptrs_buf, (n * 4) as u64, &staging, 0, 4);
    client.submit_and_wait(encoder);
    let mut total = [0i32; 1];
    client.read_buffer(&staging, &mut total)?;
    let total = usize::try_from(total[0]).map_err(|_| {
        Error::Internal(format!(
            "sparse pointer scan over {n} buckets returned a negative total {}",
            total[0]
        ))
    })?;
    Ok((ptrs, total))
}
