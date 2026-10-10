//! The grouping half of scatter_reduce: every source element's destination
//! key, then a stable LSD radix sort of the keys with the source positions as
//! values. After it, each destination's contributions sit in one run of the
//! sorted keys, in increasing source position.
//!
//! Every dispatch records into the caller's command encoder, so the whole
//! scatter_reduce reaches the queue as one submission.

use wgpu::{Buffer, CommandEncoder, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, WORKGROUP_SIZE};
use crate::error::{Error, Result};
use crate::ops::common::collapse_scatter_axes;

const KEYS_SHADER: &str = include_str!("../scatter_reduce_keys.wgsl");
const RADIX_SHADER: &str = include_str!("../scatter_reduce_radix.wgsl");

/// Axis slots the key shader walks. Must match the 8 slots of SrKeyParams.
const KEY_SLOTS: usize = 8;
/// Digits per radix pass. Must match SR_RADIX.
const RADIX: usize = 256;
/// Bits per radix pass.
const RADIX_BITS: u32 = 8;
/// Source entries per radix tile. Must match SR_TILE.
const RADIX_TILE: usize = 4096;

/// Matches SrKeyParams in scatter_reduce_keys.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KeyParams {
    n: u32,
    dim_slot: u32,
    dim_size: u32,
    invalid_key: u32,
    src_extent: [u32; KEY_SLOTS],
    dst_stride: [u32; KEY_SLOTS],
}

/// Matches SrRadixParams in scatter_reduce_radix.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RadixParams {
    n: u32,
    shift: u32,
    tiles: u32,
    _pad: u32,
}

/// Keys and source positions, sorted by key. Each run of one key lists its
/// source positions in increasing order.
pub(super) struct SortedKeys {
    pub keys: Buffer,
    pub vals: Buffer,
}

/// The index tensor's element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterIndexType {
    /// 32-bit signed indices.
    I32,
    /// 64-bit signed indices, read as two u32 words.
    I64,
}

fn limitation(reason: String) -> Error {
    Error::backend_limitation("WebGPU", "scatter_reduce", reason)
}

/// Workgroup grid for `groups` workgroups: x up to the device's per-dimension
/// limit, the rest in y. Shaders flatten it back with `num_workgroups`.
pub(super) fn grid(cache: &PipelineCache, groups: u32) -> (u32, u32) {
    let max_x = cache.device().limits().max_compute_workgroups_per_dimension;
    if groups <= max_x {
        (groups, 1)
    } else {
        (max_x, groups.div_ceil(max_x))
    }
}

pub(super) fn storage_buffer(cache: &PipelineCache, label: &'static str, words: usize) -> Buffer {
    cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (words.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

pub(super) fn uniform_buffer<T: bytemuck::Pod>(
    cache: &PipelineCache,
    queue: &Queue,
    label: &'static str,
    params: &T,
) -> Buffer {
    let buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(params));
    buffer
}

/// One dispatch of `entry_point` from module `module_key`.
#[allow(clippy::too_many_arguments)]
pub(super) fn record(
    cache: &PipelineCache,
    encoder: &mut CommandEncoder,
    module_key: &'static str,
    shader: &'static str,
    entry_point: &'static str,
    buffers: &[&Buffer],
    num_readonly_storage: u32,
    groups: (u32, u32),
) {
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: (buffers.len() - 1) as u32,
        num_uniform_buffers: 1,
        num_readonly_storage,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, buffers);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(entry_point),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, Some(&bind_group), &[]);
    pass.dispatch_workgroups(groups.0, groups.1, 1);
}

/// Records each source element's destination key and the sort of the keys.
///
/// An index outside `[0, dst_shape[dim])` gets key `dst_numel`, past every
/// real destination. `n` is `src_shape`'s element count and must be positive.
///
/// # Errors
///
/// Returns [`Error::BackendLimitation`] when the source holds more than
/// `u32::MAX / 2` elements, the destination `u32::MAX` or more, a scratch
/// buffer exceeds the device's storage-binding limit, or the shapes keep more
/// than 8 axes after collapsing.
#[allow(clippy::too_many_arguments)]
pub(super) fn record_sorted_keys(
    cache: &PipelineCache,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    indices: &Buffer,
    index_type: ScatterIndexType,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
) -> Result<SortedKeys> {
    let n: usize = src_shape.iter().product();
    let dst_numel: usize = dst_shape.iter().product();
    // I64 indices are addressed as word `2 * i + 1`, which must stay in u32.
    if n > (u32::MAX / 2) as usize {
        return Err(limitation(format!(
            "source has {n} elements; the shaders address at most {}",
            u32::MAX / 2
        )));
    }
    if dst_numel >= u32::MAX as usize {
        return Err(limitation(format!(
            "destination has {dst_numel} elements; the shaders address fewer than {}",
            u32::MAX
        )));
    }
    let limits = cache.device().limits();
    let binding_limit = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let scratch_bytes = (n as u64) * 4;
    if scratch_bytes > binding_limit {
        return Err(limitation(format!(
            "sorting {n} keys needs a {scratch_bytes}-byte buffer; the device binds at most \
             {binding_limit} bytes"
        )));
    }
    let (axes, dim_pos) = collapse_scatter_axes(src_shape, dst_shape, dim);
    if axes.len() > KEY_SLOTS {
        return Err(limitation(format!(
            "source shape {src_shape:?} against destination shape {dst_shape:?} keeps {} axes; \
             the key shader walks at most {KEY_SLOTS}",
            axes.len()
        )));
    }

    // Right-align the axes: a leading unused slot keeps extent 1, stride 0.
    let first_slot = KEY_SLOTS - axes.len();
    let mut key_params = KeyParams {
        n: n as u32,
        dim_slot: (first_slot + dim_pos) as u32,
        dim_size: dst_shape[dim] as u32,
        invalid_key: dst_numel as u32,
        src_extent: [1; KEY_SLOTS],
        dst_stride: [0; KEY_SLOTS],
    };
    for (slot, axis) in axes.iter().enumerate() {
        key_params.src_extent[first_slot + slot] = axis.src_extent as u32;
        key_params.dst_stride[first_slot + slot] = axis.dst_stride as u32;
    }

    let mut keys = storage_buffer(cache, "scatter_reduce_keys", n);
    let mut vals = storage_buffer(cache, "scatter_reduce_vals", n);
    let mut keys_alt = storage_buffer(cache, "scatter_reduce_keys_alt", n);
    let mut vals_alt = storage_buffer(cache, "scatter_reduce_vals_alt", n);
    let tiles = n.div_ceil(RADIX_TILE);
    let hist = storage_buffer(cache, "scatter_reduce_hist", RADIX * tiles);
    let totals = storage_buffer(cache, "scatter_reduce_totals", RADIX);

    let key_entry = match index_type {
        ScatterIndexType::I32 => "scatter_reduce_keys_i32",
        ScatterIndexType::I64 => "scatter_reduce_keys_i64",
    };
    let key_params_buf = uniform_buffer(cache, queue, "scatter_reduce_key_params", &key_params);
    record(
        cache,
        encoder,
        "scatter_reduce_keys",
        KEYS_SHADER,
        key_entry,
        &[indices, &keys, &vals, &key_params_buf],
        1,
        grid(cache, n.div_ceil(WORKGROUP_SIZE as usize) as u32),
    );

    // The largest key is `invalid_key`, so its bit length bounds the passes.
    let key_bits = u32::BITS - key_params.invalid_key.leading_zeros();
    let passes = key_bits.div_ceil(RADIX_BITS).max(1);
    let tile_grid = grid(cache, tiles as u32);
    for pass in 0..passes {
        let params = RadixParams {
            n: n as u32,
            shift: pass * RADIX_BITS,
            tiles: tiles as u32,
            _pad: 0,
        };
        let params_buf = uniform_buffer(cache, queue, "scatter_reduce_radix_params", &params);
        record(
            cache,
            encoder,
            "scatter_reduce_radix",
            RADIX_SHADER,
            "scatter_reduce_radix_hist",
            &[&keys, &hist, &params_buf],
            1,
            tile_grid,
        );
        record(
            cache,
            encoder,
            "scatter_reduce_radix",
            RADIX_SHADER,
            "scatter_reduce_radix_scan",
            &[&hist, &totals, &params_buf],
            0,
            (RADIX as u32, 1),
        );
        record(
            cache,
            encoder,
            "scatter_reduce_radix",
            RADIX_SHADER,
            "scatter_reduce_radix_scatter",
            &[
                &keys,
                &vals,
                &hist,
                &totals,
                &keys_alt,
                &vals_alt,
                &params_buf,
            ],
            4,
            tile_grid,
        );
        std::mem::swap(&mut keys, &mut keys_alt);
        std::mem::swap(&mut vals, &mut vals_alt);
    }

    Ok(SortedKeys { keys, vals })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses and validates `source` with naga, as pipeline creation does.
    fn validate_wgsl(name: &str, source: &str) {
        use wgpu::naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }

    #[test]
    fn the_sort_shaders_validate() {
        validate_wgsl("scatter_reduce_keys", KEYS_SHADER);
        validate_wgsl("scatter_reduce_radix", RADIX_SHADER);
    }

    #[test]
    fn key_params_match_the_wgsl_layout() {
        // Four scalars, then two arrays of two vec4<u32>.
        assert_eq!(std::mem::size_of::<KeyParams>(), 16 + 32 + 32);
        assert_eq!(std::mem::size_of::<RadixParams>(), 16);
    }

    #[test]
    fn the_tile_spans_every_round() {
        // SR_ITEMS rounds of SR_THREADS entries fill one tile.
        assert_eq!(RADIX_TILE, 16 * WORKGROUP_SIZE as usize);
    }
}
