//! Launcher for the colour kernels in color_yuv.wgsl, color_rgb24.wgsl and
//! color_pad.wgsl.
//!
//! Every operand binds as a read-write `array<u32>`, and views of one
//! storage bind once. WebGPU rejects one buffer bound both read-only and
//! read-write in a pass, and the planes of one surface share a buffer. A
//! generated prelude maps each operand to the binding of its buffer through
//! `ld_<name>` and `st_<name>`, so one kernel source serves every aliasing
//! pattern. Each pattern gets its own module, cached under a key that names
//! the pattern.

use std::fmt::Write as _;
use std::sync::LazyLock;

use wgpu::Buffer;

use super::copy::dispatch_grid;
use super::pipeline::LayoutKey;
use super::strided_common::THREAD_INDEX_WGSL;
use crate::error::{Error, Result};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::client::get_buffer;

const FLOAT_SHADER: &str = include_str!("color_float.wgsl");
const YUV_SHADER: &str = include_str!("color_yuv.wgsl");
const RGB24_SHADER: &str = include_str!("color_rgb24.wgsl");
const PAD_SHADER: &str = include_str!("color_pad.wgsl");

static YUV_BODY: LazyLock<String> =
    LazyLock::new(|| format!("{THREAD_INDEX_WGSL}{FLOAT_SHADER}{YUV_SHADER}"));

static RGB24_BODY: LazyLock<String> =
    LazyLock::new(|| format!("{THREAD_INDEX_WGSL}{FLOAT_SHADER}{RGB24_SHADER}"));

static PAD_BODY: LazyLock<String> = LazyLock::new(|| format!("{THREAD_INDEX_WGSL}{PAD_SHADER}"));

/// Colour kernel entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorEntry {
    /// `yuv_to_rgb` in color_yuv.wgsl.
    YuvToRgb,
    /// `rgb_to_yuv` in color_yuv.wgsl.
    RgbToYuv,
    /// `rgb24_to_float` in color_rgb24.wgsl.
    Rgb24ToFloat,
    /// `float_to_rgb24` in color_rgb24.wgsl.
    FloatToRgb24,
    /// `pad_packed_luma` in color_pad.wgsl.
    PadPackedLuma,
}

impl ColorEntry {
    /// WGSL entry point name.
    pub fn name(self) -> &'static str {
        match self {
            Self::YuvToRgb => "yuv_to_rgb",
            Self::RgbToYuv => "rgb_to_yuv",
            Self::Rgb24ToFloat => "rgb24_to_float",
            Self::FloatToRgb24 => "float_to_rgb24",
            Self::PadPackedLuma => "pad_packed_luma",
        }
    }

    /// Operand names, in the order [`launch_color`] takes their storages.
    pub fn operands(self) -> &'static [&'static str] {
        match self {
            Self::YuvToRgb | Self::RgbToYuv => &["rgb", "y", "u", "v"],
            Self::Rgb24ToFloat | Self::FloatToRgb24 | Self::PadPackedLuma => &["src", "dst"],
        }
    }

    /// Module key prefix, params struct name and kernel source after the
    /// prelude.
    fn module(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::YuvToRgb | Self::RgbToYuv => ("color_yuv", "YuvParams", YUV_BODY.as_str()),
            Self::Rgb24ToFloat | Self::FloatToRgb24 => {
                ("color_rgb24", "Rgb24Params", RGB24_BODY.as_str())
            }
            Self::PadPackedLuma => ("color_pad", "PadLumaParams", PAD_BODY.as_str()),
        }
    }
}

/// Binding slot of each operand: the index of the first operand with the
/// same storage.
fn binding_slots(storages: &[u64]) -> (Vec<u64>, Vec<usize>) {
    let mut distinct: Vec<u64> = Vec::with_capacity(storages.len());
    let slots = storages
        .iter()
        .map(|&s| match distinct.iter().position(|&d| d == s) {
            Some(slot) => slot,
            None => {
                distinct.push(s);
                distinct.len() - 1
            }
        })
        .collect();
    (distinct, slots)
}

/// Bindings and operand accessors for `slots` over `bindings` buffers.
fn prelude(names: &[&str], slots: &[usize], bindings: usize, params: &str) -> String {
    let mut out = String::new();
    for b in 0..bindings {
        let _ = writeln!(
            out,
            "@group(0) @binding({b}) var<storage, read_write> buf{b}: array<u32>;"
        );
    }
    let _ = writeln!(
        out,
        "@group(0) @binding({bindings}) var<uniform> params: {params};"
    );
    for (name, slot) in names.iter().zip(slots) {
        let _ = writeln!(
            out,
            "fn ld_{name}(i: u32) -> u32 {{ return buf{slot}[i]; }}\n\
             fn st_{name}(i: u32, v: u32) {{ buf{slot}[i] = v; }}"
        );
    }
    out
}

/// Launch `entry` over `total` invocations.
///
/// `storages` holds the storage handle of each operand of
/// [`ColorEntry::operands`], in order. `params` is a uniform buffer holding
/// the entry's params struct.
///
/// # Errors
///
/// `Error::Internal` when the operand count is wrong or a storage is not in
/// the wgpu buffer registry.
pub fn launch_color(
    client: &WgpuClient,
    entry: ColorEntry,
    storages: &[u64],
    params: &Buffer,
    total: usize,
) -> Result<()> {
    let op = entry.name();
    let names = entry.operands();
    if storages.len() != names.len() {
        return Err(Error::Internal(format!(
            "{op}: launch_color received {} operands, expected {} ({names:?})",
            storages.len(),
            names.len()
        )));
    }
    if total == 0 {
        return Ok(());
    }
    let (distinct, slots) = binding_slots(storages);
    let buffers = distinct
        .iter()
        .map(|&handle| {
            get_buffer(handle).ok_or_else(|| {
                Error::Internal(format!(
                    "{op}: buffer {handle} is not in the wgpu buffer registry"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let (prefix, params_struct, body) = entry.module();
    let mut key = String::from(prefix);
    for slot in &slots {
        let _ = write!(key, "_{slot}");
    }
    let source = prelude(names, &slots, distinct.len(), params_struct) + body;
    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module_from_source(&key, &source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: distinct.len() as u32,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(&key, op, &module, &layout);
    let mut bound: Vec<&Buffer> = buffers.iter().map(|b| &**b).collect();
    bound.push(params);
    let bind_group = cache.create_bind_group(&layout, &bound);

    let (x, y) = dispatch_grid(total);
    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
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
    fn views_of_one_storage_share_a_binding() {
        let (distinct, slots) = binding_slots(&[7, 3, 3, 3]);
        assert_eq!(distinct, [7, 3]);
        assert_eq!(slots, [0, 1, 1, 1]);
        let (distinct, slots) = binding_slots(&[5, 5]);
        assert_eq!((distinct.len(), slots), (1, vec![0, 0]));
    }

    #[test]
    fn prelude_maps_every_operand_to_its_binding() {
        let text = prelude(&["rgb", "y", "u", "v"], &[0, 1, 2, 2], 3, "YuvParams");
        assert!(text.contains("@binding(2) var<storage, read_write> buf2"));
        assert!(text.contains("@binding(3) var<uniform> params: YuvParams;"));
        assert!(text.contains("fn ld_v(i: u32) -> u32 { return buf2[i]; }"));
        assert!(text.contains("fn st_y(i: u32, v: u32) { buf1[i] = v; }"));
    }

    #[test]
    fn every_entry_point_is_in_its_shader() {
        for entry in [
            ColorEntry::YuvToRgb,
            ColorEntry::RgbToYuv,
            ColorEntry::Rgb24ToFloat,
            ColorEntry::FloatToRgb24,
            ColorEntry::PadPackedLuma,
        ] {
            let (_, params, body) = entry.module();
            assert!(body.contains(&format!("fn {}(", entry.name())));
            assert!(body.contains(&format!("struct {params} {{")));
            for name in entry.operands() {
                assert!(body.contains(&format!("_{name}(")), "{name} unused");
            }
        }
    }
}
