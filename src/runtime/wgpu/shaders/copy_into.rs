//! Strided-to-strided copy on WebGPU, the kernel behind `copy_into` with a
//! non-contiguous destination.
//!
//! WGSL storage buffers are `array<u32>`, so a thread moves whole 4-byte
//! words. The kernel needs the element size and both byte offsets to be
//! multiples of 4, which holds for the 32-bit dtypes WebGPU supports. Any
//! other size returns `BackendLimitation`: a sub-word destination write would
//! race with the neighbouring elements that share its word.
//!
//! Two entry points cover the buffer layouts:
//!
//! - `strided_copy_into` binds the source and the destination as two buffers.
//! - `strided_copy_into_shared` binds one read-write buffer for both. A
//!   source and a destination in the same buffer cannot bind as read-only
//!   plus read-write in one pass. `copy_into` has already rejected
//!   overlapping views, so the two views touch disjoint words.
//!
//! Params layout: `[count, ndim, units, src_base, dst_base, 0, 0, 0,
//! shape[8], src_strides[8], dst_strides[8]]`. `units` is the element size in
//! words. `src_base` and `dst_base` are in words. Strides are in elements.

use std::sync::LazyLock;

use wgpu::{Buffer, BufferUsages};

use super::copy::{MAX_DIMS, dispatch_grid};
use super::pipeline::LayoutKey;
use super::strided_common::{
    THREAD_INDEX_WGSL, limitation as limitation_for, shape_words, stride_words,
    to_u32 as to_u32_for,
};
use crate::error::Result;
use crate::runtime::wgpu::client::WgpuClient;

/// Words in the params header before the shape.
const HEADER: usize = 8;

/// Offset function shared by both variants.
const OFFSETS_FN: &str = r#"
// Element offsets of linear index `linear` in the source and the destination.
// Strides can be negative. u32 arithmetic wraps, so the sums are exact
// modulo 2^32 and land in range once the base offsets are added.
fn element_offsets(linear: u32) -> vec2<u32> {
    let ndim = params[1];
    var remaining = linear;
    var src_off: u32 = 0u;
    var dst_off: u32 = 0u;
    for (var d: i32 = i32(ndim) - 1; d >= 0; d = d - 1) {
        let dim_size = params[8u + u32(d)];
        let idx = remaining % dim_size;
        remaining = remaining / dim_size;
        src_off = src_off + idx * params[16u + u32(d)];
        dst_off = dst_off + idx * params[24u + u32(d)];
    }
    return vec2<u32>(src_off, dst_off);
}
"#;

const TWO_BUFFER_BINDINGS: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
"#;

const ONE_BUFFER_BINDINGS: &str = r#"
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read_write> buf: array<u32>;
"#;

/// Kernel body. `SRC` and `DST` name the buffers each variant reads and writes.
const KERNEL: &str = r#"
@compute @workgroup_size(256)
fn ENTRY(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = thread_index(gid);
    if (t >= params[0]) {
        return;
    }
    let offsets = element_offsets(t);
    let units = params[2];
    let src_unit = params[3] + offsets.x * units;
    let dst_unit = params[4] + offsets.y * units;
    for (var i: u32 = 0u; i < units; i = i + 1u) {
        DST[dst_unit + i] = SRC[src_unit + i];
    }
}
"#;

static TWO_BUFFER_SOURCE: LazyLock<String> = LazyLock::new(|| {
    let kernel = KERNEL
        .replace("ENTRY", "strided_copy_into")
        .replace("SRC", "src")
        .replace("DST", "dst");
    format!("{TWO_BUFFER_BINDINGS}{THREAD_INDEX_WGSL}{OFFSETS_FN}{kernel}")
});

static ONE_BUFFER_SOURCE: LazyLock<String> = LazyLock::new(|| {
    let kernel = KERNEL
        .replace("ENTRY", "strided_copy_into_shared")
        .replace("SRC", "buf")
        .replace("DST", "buf");
    format!("{ONE_BUFFER_BINDINGS}{THREAD_INDEX_WGSL}{OFFSETS_FN}{kernel}")
});

/// Operation name in this file's errors.
const OPERATION: &str = "copy_into";

fn limitation(reason: String) -> crate::error::Error {
    limitation_for(OPERATION, reason)
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    to_u32_for(OPERATION, value, what)
}

/// Copy a strided source view into a strided destination view.
///
/// `shared` is true when `src` and `dst` are the same buffer. Strides are in
/// elements and can be negative. The byte offsets locate each view's first
/// element in its buffer.
///
/// # Errors
///
/// `BackendLimitation` when the rank exceeds [`MAX_DIMS`], when the element
/// size or an offset is not a multiple of 4 bytes, or when a count, offset or
/// stride does not fit the kernel's 32-bit addressing.
pub fn launch_strided_copy_into(
    client: &WgpuClient,
    src: &Buffer,
    dst: &Buffer,
    shared: bool,
    shape: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    elem_size: usize,
    src_byte_offset: usize,
    dst_byte_offset: usize,
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
    if !elem_size.is_multiple_of(4)
        || !src_byte_offset.is_multiple_of(4)
        || !dst_byte_offset.is_multiple_of(4)
    {
        return Err(limitation(format!(
            "a strided destination needs 4-byte aligned elements and offsets, got element \
             size {elem_size}, source offset {src_byte_offset}, destination offset \
             {dst_byte_offset}"
        )));
    }
    let total_bytes = numel
        .checked_mul(elem_size)
        .ok_or_else(|| limitation(format!("{numel} elements of {elem_size} bytes overflow")))?;
    to_u32(total_bytes, "byte count")?;

    let mut params = Vec::with_capacity(HEADER + 3 * MAX_DIMS);
    params.push(to_u32(numel, "element count")?);
    params.push(shape.len() as u32);
    params.push(to_u32(elem_size / 4, "element size")?);
    params.push(to_u32(src_byte_offset / 4, "source offset")?);
    params.push(to_u32(dst_byte_offset / 4, "destination offset")?);
    params.resize(HEADER, 0);

    let mut dims = [0u32; MAX_DIMS];
    shape_words(OPERATION, shape, &mut dims)?;
    let mut src_steps = [0u32; MAX_DIMS];
    let mut dst_steps = [0u32; MAX_DIMS];
    stride_words(OPERATION, src_strides, &mut src_steps)?;
    stride_words(OPERATION, dst_strides, &mut dst_steps)?;
    params.extend_from_slice(&dims);
    params.extend_from_slice(&src_steps);
    params.extend_from_slice(&dst_steps);

    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("strided_copy_into_params"),
        size: (params.len() * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client.write_buffer(&params_buffer, &params);

    let cache = client.pipeline_cache();
    let (module_key, entry_point, body, layout_key) = if shared {
        (
            "strided_copy_into_shared",
            "strided_copy_into_shared",
            &*ONE_BUFFER_SOURCE,
            LayoutKey {
                num_storage_buffers: 2,
                num_uniform_buffers: 0,
                num_readonly_storage: 1,
            },
        )
    } else {
        (
            "strided_copy_into",
            "strided_copy_into",
            &*TWO_BUFFER_SOURCE,
            LayoutKey {
                num_storage_buffers: 3,
                num_uniform_buffers: 0,
                num_readonly_storage: 2,
            },
        )
    };
    let module = cache.get_or_create_module(module_key, body);
    let layout = cache.get_or_create_layout(layout_key);
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = if shared {
        cache.create_bind_group(&layout, &[&params_buffer, dst])
    } else {
        cache.create_bind_group(&layout, &[src, &params_buffer, dst])
    };

    let (x, y) = dispatch_grid(numel);
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
    fn workgroup_size_matches_the_wgsl_constant() {
        assert_eq!(super::super::pipeline::WORKGROUP_SIZE, 256);
        assert!(THREAD_INDEX_WGSL.contains("WORKGROUP_SIZE: u32 = 256u"));
    }

    #[test]
    fn variants_name_their_own_entry_points() {
        assert!(TWO_BUFFER_SOURCE.contains("fn strided_copy_into("));
        assert!(ONE_BUFFER_SOURCE.contains("fn strided_copy_into_shared("));
        assert!(!ONE_BUFFER_SOURCE.contains("SRC"));
        assert!(!ONE_BUFFER_SOURCE.contains("DST"));
    }
}
