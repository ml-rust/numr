//! Strided-to-contiguous copy on WebGPU, the kernel behind
//! `Tensor::contiguous()`.
//!
//! WGSL storage buffers are `array<u32>`, so a thread addresses 4-byte words.
//! Two kernels cover every element size:
//!
//! - `strided_copy` moves whole words. It runs when the element size and the
//!   source byte offset are both multiples of 4.
//! - `strided_copy_bytes` builds each output word from 4 source bytes. It runs
//!   for 1- and 2-byte elements (U8, I8, Bool, F16, BF16, quantized block
//!   bytes), odd element sizes, and unaligned source offsets.
//!
//! The byte kernel writes whole output words. A tensor buffer is allocated in
//! 4-byte units, so the final word's padding bytes belong to the destination
//! and receive zeros.
//!
//! A dispatch dimension holds at most 65535 workgroups. Both kernels spread
//! their threads over a 2-D grid, so a copy of any size up to 4 GiB runs.

use std::sync::LazyLock;

use wgpu::{Buffer, BufferUsages};

use super::pipeline::{LayoutKey, WORKGROUP_SIZE};
use crate::error::{Error, Result};
use crate::runtime::wgpu::client::WgpuClient;

/// Maximum tensor rank the kernels accept.
pub const MAX_DIMS: usize = 8;

/// Workgroups per row of the 2-D dispatch grid.
const GRID_X: u32 = 65535;

/// Shared WGSL prelude: bindings, parameter access and the strided offset.
///
/// Params layout: `[count, ndim, elem_size, src_offset, shape[8], strides[8]]`.
/// `strided_copy` reads `elem_size` and `src_offset` in 4-byte units and
/// `count` in elements. `strided_copy_bytes` reads all three in bytes.
const PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

const GRID_X: u32 = 65535u;
const WORKGROUP_SIZE: u32 = 256u;

fn thread_index(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * GRID_X * WORKGROUP_SIZE;
}

// Element offset of linear index `linear` in the source, in elements.
// Strides can be negative. u32 arithmetic wraps, so the sum is exact
// modulo 2^32 and lands in range once the base offset is added.
fn strided_elem_offset(linear: u32) -> u32 {
    let ndim = params[1];
    var remaining = linear;
    var offset: u32 = 0u;
    for (var d: i32 = i32(ndim) - 1; d >= 0; d = d - 1) {
        let dim_size = params[4u + u32(d)];
        let idx = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + idx * params[12u + u32(d)];
    }
    return offset;
}
"#;

/// Word kernel: one thread copies one element of `elem_size` words.
const WORD_KERNEL: &str = r#"
@compute @workgroup_size(256)
fn strided_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    let numel = params[0];
    if (t >= numel) {
        return;
    }
    let units = params[2];
    let src_unit = params[3] + strided_elem_offset(t) * units;
    let dst_unit = t * units;
    for (var i: u32 = 0u; i < units; i = i + 1u) {
        dst[dst_unit + i] = src[src_unit + i];
    }
}
"#;

/// Byte kernel: one thread writes one output word, gathered byte by byte.
const BYTE_KERNEL: &str = r#"
@compute @workgroup_size(256)
fn strided_copy_bytes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = thread_index(gid);
    let total_bytes = params[0];
    let first = word * 4u;
    if (first >= total_bytes) {
        return;
    }
    let elem_size = params[2];
    let src_offset = params[3];
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let b = first + k;
        if (b >= total_bytes) {
            break;
        }
        let elem = b / elem_size;
        let within = b - elem * elem_size;
        let src_byte = src_offset + strided_elem_offset(elem) * elem_size + within;
        let v = (src[src_byte >> 2u] >> ((src_byte & 3u) * 8u)) & 0xffu;
        packed = packed | (v << (k * 8u));
    }
    dst[word] = packed;
}
"#;

static WORD_SOURCE: LazyLock<String> = LazyLock::new(|| format!("{PRELUDE}{WORD_KERNEL}"));
static BYTE_SOURCE: LazyLock<String> = LazyLock::new(|| format!("{PRELUDE}{BYTE_KERNEL}"));

/// Which kernel a copy needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyKernel {
    /// Whole 4-byte words per element.
    Words,
    /// Single bytes gathered into output words.
    Bytes,
}

impl CopyKernel {
    /// The kernel for an element size and source byte offset.
    pub fn select(elem_size: usize, src_byte_offset: usize) -> Self {
        if elem_size.is_multiple_of(4) && src_byte_offset.is_multiple_of(4) {
            Self::Words
        } else {
            Self::Bytes
        }
    }
}

/// The 2-D workgroup grid for `threads` threads.
pub fn dispatch_grid(threads: usize) -> (u32, u32) {
    let groups = threads.div_ceil(WORKGROUP_SIZE as usize);
    let x = groups.min(GRID_X as usize);
    let y = groups.div_ceil(x.max(1));
    (x as u32, y as u32)
}

fn limitation(reason: String) -> Error {
    Error::BackendLimitation {
        backend: "wgpu",
        operation: "copy_strided",
        reason,
    }
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        limitation(format!(
            "{what} {value} exceeds the 32-bit addressing of the copy kernel"
        ))
    })
}

/// Copy a strided source view into the contiguous buffer `dst`.
///
/// `strides` are in elements. `src_byte_offset` is the view's first byte.
///
/// # Errors
///
/// `BackendLimitation` when the rank exceeds [`MAX_DIMS`], or when a byte
/// count, offset or stride does not fit the kernel's 32-bit addressing.
pub fn launch_strided_copy(
    client: &WgpuClient,
    src: &Buffer,
    dst: &Buffer,
    shape: &[usize],
    strides: &[isize],
    elem_size: usize,
    src_byte_offset: usize,
) -> Result<()> {
    if shape.len() > MAX_DIMS {
        return Err(limitation(format!(
            "rank {} exceeds the maximum of {MAX_DIMS} dimensions",
            shape.len()
        )));
    }
    let numel: usize = shape.iter().product();
    if numel == 0 || elem_size == 0 {
        return Ok(());
    }
    let total_bytes = numel
        .checked_mul(elem_size)
        .ok_or_else(|| limitation(format!("{numel} elements of {elem_size} bytes overflow")))?;
    to_u32(total_bytes, "byte count")?;
    to_u32(src_byte_offset, "source byte offset")?;

    let kernel = CopyKernel::select(elem_size, src_byte_offset);
    let (count, size, offset, threads) = match kernel {
        CopyKernel::Words => (numel, elem_size / 4, src_byte_offset / 4, numel),
        CopyKernel::Bytes => (
            total_bytes,
            elem_size,
            src_byte_offset,
            total_bytes.div_ceil(4),
        ),
    };

    let mut params = Vec::with_capacity(4 + 2 * MAX_DIMS);
    params.push(to_u32(count, "element count")?);
    params.push(shape.len() as u32);
    params.push(to_u32(size, "element size")?);
    params.push(to_u32(offset, "source offset")?);
    let mut dims = [0u32; MAX_DIMS];
    let mut steps = [0u32; MAX_DIMS];
    for (d, (&dim, &stride)) in shape.iter().zip(strides).enumerate() {
        dims[d] = to_u32(dim, "dimension")?;
        let stride = i32::try_from(stride)
            .map_err(|_| limitation(format!("stride {stride} does not fit 32 bits")))?;
        // The kernel adds strides as wrapping u32, which equals i32 addition.
        steps[d] = stride as u32;
    }
    params.extend_from_slice(&dims);
    params.extend_from_slice(&steps);

    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("strided_copy_params"),
        size: (params.len() * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client.write_buffer(&params_buffer, &params);

    let (module_key, entry_point, body) = match kernel {
        CopyKernel::Words => ("strided_copy_words", "strided_copy", &*WORD_SOURCE),
        CopyKernel::Bytes => ("strided_copy_bytes", "strided_copy_bytes", &*BYTE_SOURCE),
    };
    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(module_key, body);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 0,
        num_readonly_storage: 2,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[src, &params_buffer, dst]);

    let (x, y) = dispatch_grid(threads);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(module_key),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(module_key),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
    client.submit_and_wait(encoder);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_choice_follows_alignment() {
        assert_eq!(CopyKernel::select(4, 0), CopyKernel::Words);
        assert_eq!(CopyKernel::select(8, 16), CopyKernel::Words);
        assert_eq!(CopyKernel::select(1, 0), CopyKernel::Bytes);
        assert_eq!(CopyKernel::select(2, 4), CopyKernel::Bytes);
        assert_eq!(CopyKernel::select(4, 2), CopyKernel::Bytes);
    }

    #[test]
    fn grid_covers_every_thread() {
        assert_eq!(dispatch_grid(1), (1, 1));
        assert_eq!(dispatch_grid(256), (1, 1));
        assert_eq!(dispatch_grid(257), (2, 1));
        let threads = 65535 * 256 + 1;
        let (x, y) = dispatch_grid(threads);
        assert_eq!((x, y), (65535, 2));
        assert!(x as usize * y as usize * 256 >= threads);
    }
}
