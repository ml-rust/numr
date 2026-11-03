//! Launcher for the `resize` pass kernel in image_resize.wgsl. F32 only.
//!
//! Two module variants cover the buffer layouts:
//!
//! - `image_resize` binds the source read-only and the destination
//!   read-write as two buffers.
//! - `image_resize_shared` binds one read-write buffer for both. WebGPU
//!   rejects one buffer bound read-only and read-write in one pass. The
//!   caller checked that the two views share no element.

use std::sync::LazyLock;

use wgpu::Buffer;

use super::copy::dispatch_grid;
use super::pipeline::LayoutKey;
use super::strided_common::THREAD_INDEX_WGSL;
use crate::error::Result;
use crate::runtime::wgpu::WgpuClient;

const RESIZE_SHADER: &str = include_str!("image_resize.wgsl");

/// WGSL entry point of the pass kernel.
const ENTRY: &str = "resize_pass";

const TWO_BUFFER_PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: ResizeParams;

fn load_src(i: u32) -> f32 {
    return src[i];
}

fn store_dst(i: u32, v: f32) {
    dst[i] = v;
}
"#;

const ONE_BUFFER_PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> params: ResizeParams;

fn load_src(i: u32) -> f32 {
    return buf[i];
}

fn store_dst(i: u32, v: f32) {
    buf[i] = v;
}
"#;

static TWO_BUFFER_SOURCE: LazyLock<String> =
    LazyLock::new(|| format!("{TWO_BUFFER_PRELUDE}{THREAD_INDEX_WGSL}{RESIZE_SHADER}"));

static ONE_BUFFER_SOURCE: LazyLock<String> =
    LazyLock::new(|| format!("{ONE_BUFFER_PRELUDE}{THREAD_INDEX_WGSL}{RESIZE_SHADER}"));

/// Launch one pass over `total` destination elements.
///
/// `shared` is true when `src` and `dst` are the same buffer; the caller
/// passes that buffer as both. `params` is a uniform buffer holding
/// `ResizeParams`. The caller checked the dtype: F32.
pub fn launch_resize_pass(
    client: &WgpuClient,
    src: &Buffer,
    dst: &Buffer,
    shared: bool,
    params: &Buffer,
    total: usize,
) -> Result<()> {
    if total == 0 {
        return Ok(());
    }
    let cache = client.pipeline_cache();
    let (module_key, source, layout_key) = if shared {
        (
            "image_resize_shared",
            &*ONE_BUFFER_SOURCE,
            LayoutKey {
                num_storage_buffers: 1,
                num_uniform_buffers: 1,
                num_readonly_storage: 0,
            },
        )
    } else {
        (
            "image_resize",
            &*TWO_BUFFER_SOURCE,
            LayoutKey {
                num_storage_buffers: 2,
                num_uniform_buffers: 1,
                num_readonly_storage: 1,
            },
        )
    };
    let module = cache.get_or_create_module(module_key, source);
    let layout = cache.get_or_create_layout(layout_key);
    let pipeline = cache.get_or_create_pipeline(module_key, ENTRY, &module, &layout);
    let bind_group = if shared {
        cache.create_bind_group(&layout, &[dst, params])
    } else {
        cache.create_bind_group(&layout, &[src, dst, params])
    };

    let (x, y) = dispatch_grid(total);
    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(ENTRY) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(ENTRY),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_bind_their_own_buffers() {
        assert!(TWO_BUFFER_SOURCE.contains("var<storage, read> src"));
        assert!(ONE_BUFFER_SOURCE.contains("var<storage, read_write> buf"));
        assert!(!ONE_BUFFER_SOURCE.contains("var<storage, read> src"));
        assert!(RESIZE_SHADER.contains(&format!("fn {ENTRY}(")));
    }
}
