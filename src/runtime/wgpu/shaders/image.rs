//! Launcher for the image gather kernels in image.wgsl.
//!
//! Storage buffers bind as `array<u32>`, so the kernels move 32-bit words and
//! serve F32, I32 and U32 alike. Two module variants cover the buffer layouts:
//!
//! - `image_gather` binds the source read-only and the destination
//!   read-write as two buffers.
//! - `image_gather_shared` binds one read-write buffer for both. WebGPU
//!   rejects one buffer bound read-only and read-write in one pass. The
//!   caller checked that the two views share no element.

use std::sync::LazyLock;

use wgpu::Buffer;

use super::copy::dispatch_grid;
use super::pipeline::LayoutKey;
use super::strided_common::THREAD_INDEX_WGSL;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::WgpuClient;

const IMAGE_SHADER: &str = include_str!("image.wgsl");

const TWO_BUFFER_PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: ImageParams;

fn load_src(i: u32) -> u32 {
    return src[i];
}

fn store_dst(i: u32, v: u32) {
    dst[i] = v;
}
"#;

const ONE_BUFFER_PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> params: ImageParams;

fn load_src(i: u32) -> u32 {
    return buf[i];
}

fn store_dst(i: u32, v: u32) {
    buf[i] = v;
}
"#;

static TWO_BUFFER_SOURCE: LazyLock<String> =
    LazyLock::new(|| format!("{TWO_BUFFER_PRELUDE}{THREAD_INDEX_WGSL}{IMAGE_SHADER}"));

static ONE_BUFFER_SOURCE: LazyLock<String> =
    LazyLock::new(|| format!("{ONE_BUFFER_PRELUDE}{THREAD_INDEX_WGSL}{IMAGE_SHADER}"));

/// Gather kernel entry point in image.wgsl.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageEntry {
    /// `pixel_shuffle`.
    PixelShuffle,
    /// `pixel_unshuffle`.
    PixelUnshuffle,
    /// `upsample_nearest2d`.
    UpsampleNearest2d,
}

impl ImageEntry {
    /// WGSL entry point name, which is also the op name.
    pub fn name(self) -> &'static str {
        match self {
            Self::PixelShuffle => "pixel_shuffle",
            Self::PixelUnshuffle => "pixel_unshuffle",
            Self::UpsampleNearest2d => "upsample_nearest2d",
        }
    }
}

/// Launch one gather kernel over `total` output elements.
///
/// `shared` is true when `src` and `dst` are the same buffer; the caller
/// passes that buffer as both. `params` is a uniform buffer holding
/// `ImageParams`.
///
/// # Errors
///
/// `Error::UnsupportedDType` for a dtype other than F32, I32 or U32.
#[allow(clippy::too_many_arguments)]
pub fn launch_image(
    client: &WgpuClient,
    entry: ImageEntry,
    dtype: DType,
    src: &Buffer,
    dst: &Buffer,
    shared: bool,
    params: &Buffer,
    total: usize,
) -> Result<()> {
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: entry.name(),
        });
    }
    if total == 0 {
        return Ok(());
    }

    let cache = client.pipeline_cache();
    let (module_key, source, layout_key) = if shared {
        (
            "image_gather_shared",
            &*ONE_BUFFER_SOURCE,
            LayoutKey {
                num_storage_buffers: 1,
                num_uniform_buffers: 1,
                num_readonly_storage: 0,
            },
        )
    } else {
        (
            "image_gather",
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
    let pipeline = cache.get_or_create_pipeline(module_key, entry.name(), &module, &layout);
    let bind_group = if shared {
        cache.create_bind_group(&layout, &[dst, params])
    } else {
        cache.create_bind_group(&layout, &[src, dst, params])
    };

    let (x, y) = dispatch_grid(total);
    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(entry.name()),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry.name()),
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
    }

    #[test]
    fn every_entry_point_is_in_the_shader() {
        for entry in [
            ImageEntry::PixelShuffle,
            ImageEntry::PixelUnshuffle,
            ImageEntry::UpsampleNearest2d,
        ] {
            assert!(IMAGE_SHADER.contains(&format!("fn {}(", entry.name())));
        }
    }
}
