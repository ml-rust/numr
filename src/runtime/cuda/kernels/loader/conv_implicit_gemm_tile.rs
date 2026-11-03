//! Accumulator, block tile, staging variant and launch geometry for the
//! implicit-GEMM conv2d.
//!
//! `kernels/conv_implicit_gemm.cu` instantiates each accumulator with block
//! tiles, each with three staging variants, and names the symbol after all
//! three. The GEMM view is `M = N * H_out * W_out` rows by `C_out` columns.

use crate::dtype::DType;

use super::launch_dims::{LaunchConfig, MAX_GRID_DIM_X, MAX_GRID_DIM_YZ};
use super::names::dtype_suffix;

/// Block tile of one implicit-GEMM conv kernel. Every tile has 4 warps.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum ConvTile {
    /// 128x64, 2x2 warp grid, 64x32 warp tile. F32 accumulator only.
    Tile128x64,
    /// 256x32, 4x1 warp grid, 64x32 warp tile, for `C_out <= 32`.
    Tile256x32,
    /// 256x64, 4x1 warp grid, 64x64 warp tile. F16 accumulator only.
    Tile256x64,
}

impl ConvTile {
    /// Output pixels one block computes.
    const fn tile_m(self) -> u32 {
        match self {
            Self::Tile256x32 | Self::Tile256x64 => 256,
            Self::Tile128x64 => 128,
        }
    }

    /// Output channels one block computes.
    const fn tile_n(self) -> u32 {
        match self {
            Self::Tile128x64 | Self::Tile256x64 => 64,
            Self::Tile256x32 => 32,
        }
    }

    /// Threads per block. Equals `ConvTileCfg::THREADS` of the instantiation.
    const fn threads(self) -> u32 {
        128
    }

    const fn suffix(self) -> &'static str {
        match self {
            Self::Tile128x64 => "128x64",
            Self::Tile256x32 => "256x32",
            Self::Tile256x64 => "256x64",
        }
    }

    /// Output patch `(rows, columns)` of one halo-staging block. Equals
    /// `ConvTileCfg::PATCH_H` and `PATCH_W` of the instantiation.
    const fn patch(self) -> (usize, usize) {
        match self {
            Self::Tile128x64 => (8, 16),
            Self::Tile256x32 | Self::Tile256x64 => (8, 32),
        }
    }

    /// Most halo pixels one halo-staging block holds. Equals
    /// `ConvTileCfg::HALO_PIX` of the instantiation.
    const fn halo_capacity(self) -> usize {
        match self {
            Self::Tile128x64 => 192,
            Self::Tile256x32 | Self::Tile256x64 => 352,
        }
    }
}

/// Output extent and kernel window of one launch.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) struct ConvWindow {
    pub batch: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
}

/// Input pixels under one output patch of `tile`, halo included, or `None`
/// on overflow:
/// `((ph - 1) * stride_h + (kh - 1) * dil_h + 1) * ((pw - 1) * stride_w + (kw - 1) * dil_w + 1)`.
fn halo_pixels(tile: ConvTile, w: &ConvWindow) -> Option<usize> {
    let (ph, pw) = tile.patch();
    let extent = |patch: usize, stride: usize, kernel: usize, dil: usize| {
        (patch - 1)
            .checked_mul(stride)?
            .checked_add(kernel.checked_sub(1)?.checked_mul(dil)?)?
            .checked_add(1)
    };
    let h = extent(ph, w.stride.0, w.kernel.0, w.dilation.0)?;
    let wd = extent(pw, w.stride.1, w.kernel.1, w.dilation.1)?;
    h.checked_mul(wd)
}

/// Operand staging variant of one implicit-GEMM conv kernel.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum ConvStaging {
    /// A block owns a 2D output patch. Each 32-channel chunk of the input
    /// patch plus halo is staged once with 16-byte loads and serves every
    /// tap. K runs channel-chunk-major.
    Halo,
    /// 16-byte gathers per tap, K in tap-major order.
    Vector,
    /// 16-bit gathers, K in flat `(kh, kw, ci)` order.
    Scalar,
}

impl ConvStaging {
    const fn suffix(self) -> &'static str {
        match self {
            Self::Halo => "halo",
            Self::Vector => "vec",
            Self::Scalar => "scalar",
        }
    }
}

/// Symbol of one instantiation: `conv2d_implicit_gemm_{acc}_{tile}_{staging}`.
///
/// `{acc}` is the dtype suffix for the F32 accumulator. It is `f16acc` when
/// `half_accumulate` is set, which the launcher allows for F16 only.
pub(super) fn conv_kernel_name(
    dtype: DType,
    half_accumulate: bool,
    tile: ConvTile,
    staging: ConvStaging,
) -> String {
    let acc = if half_accumulate {
        "f16acc"
    } else {
        dtype_suffix(dtype)
    };
    format!(
        "conv2d_implicit_gemm_{acc}_{}_{}",
        tile.suffix(),
        staging.suffix()
    )
}

/// Pick the staging variant for `tile`.
///
/// The halo and vector variants read 8 channels of one pixel per 16-byte
/// load. That needs `C_in % 8 == 0`, a 16-byte aligned input and weight,
/// and input N/H/W element strides that are multiples of 8. Strides are
/// never negative here; `ops/cuda/conv_fused_native.rs` refuses those views.
///
/// - Halo: the 16-byte conditions hold and the halo of one output patch
///   fits the tile's halo buffer ([`ConvTile::halo_capacity`]). A 3x3
///   stride-1 dilation-1 kernel always fits, and so does a 1x1 stride-1
///   kernel. Large strides, dilations and kernels do not.
/// - Vector: the 16-byte conditions hold and the halo does not fit.
/// - Scalar: every other shape.
pub(super) fn select_conv_staging(
    c_in: usize,
    input_ptr: u64,
    input_strides: [i64; 3],
    weight_ptr: u64,
    tile: ConvTile,
    window: &ConvWindow,
) -> ConvStaging {
    let aligned = |ptr: u64| ptr.is_multiple_of(16);
    let wide = c_in.is_multiple_of(8)
        && aligned(input_ptr)
        && aligned(weight_ptr)
        && input_strides.iter().all(|s| s % 8 == 0);
    if !wide {
        ConvStaging::Scalar
    } else if halo_pixels(tile, window).is_some_and(|px| px <= tile.halo_capacity()) {
        ConvStaging::Halo
    } else {
        ConvStaging::Vector
    }
}

/// Pick the block tile for `c_out` channels and the accumulator.
///
/// - `C_out <= 32`: 256x32. A 64-wide tile leaves half of its columns empty.
/// - Otherwise 128x64 for the F32 accumulator, or 256x64 for the F16
///   accumulator. Half-precision accumulators take half the registers, so
///   the 64x64 warp tile fits. Channels past 64 go over grid y.
pub(super) fn select_conv_tile(c_out: usize, half_accumulate: bool) -> ConvTile {
    if c_out <= 32 {
        ConvTile::Tile256x32
    } else if half_accumulate {
        ConvTile::Tile256x64
    } else {
        ConvTile::Tile128x64
    }
}

/// Grid and block for one launch: pixels on grid x, channels on grid y.
///
/// Grid x counts `tile_m`-row tiles of `M = N * H_out * W_out`, or for
/// [`ConvStaging::Halo`] the output patches of every image, in
/// `(n, patch row, patch column)` order. Pixels go on x because a large
/// frame can need more than 65535 blocks. Returns `None` when either
/// extent passes the CUDA grid limit or is 0.
pub(super) fn conv_launch_config(
    window: &ConvWindow,
    c_out: usize,
    tile: ConvTile,
    staging: ConvStaging,
) -> Option<LaunchConfig> {
    let grid_x = if staging == ConvStaging::Halo {
        let (ph, pw) = tile.patch();
        (window.batch as u64)
            .checked_mul((window.out_h as u64).div_ceil(ph as u64))?
            .checked_mul((window.out_w as u64).div_ceil(pw as u64))?
    } else {
        (window.batch as u64)
            .checked_mul(window.out_h as u64)?
            .checked_mul(window.out_w as u64)?
            .div_ceil(u64::from(tile.tile_m()))
    };
    let grid_y = (c_out as u64).div_ceil(u64::from(tile.tile_n()));
    if grid_x == 0 || grid_y == 0 {
        return None;
    }
    if grid_x > u64::from(MAX_GRID_DIM_X) || grid_y > u64::from(MAX_GRID_DIM_YZ) {
        return None;
    }
    Some(LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (tile.threads(), 1, 1),
        // Every buffer of the kernel is static shared memory.
        shared_mem_bytes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_channels_take_256x32() {
        for c_out in [1, 3, 16, 32] {
            assert_eq!(select_conv_tile(c_out, false), ConvTile::Tile256x32);
            assert_eq!(select_conv_tile(c_out, true), ConvTile::Tile256x32);
        }
    }

    #[test]
    fn wide_channels_take_128x64() {
        for c_out in [33, 48, 64, 96, 160] {
            assert_eq!(select_conv_tile(c_out, false), ConvTile::Tile128x64);
        }
    }

    #[test]
    fn wide_channels_with_half_accumulation_take_256x64() {
        for c_out in [33, 48, 64, 96, 160] {
            assert_eq!(select_conv_tile(c_out, true), ConvTile::Tile256x64);
        }
    }

    /// `batch` images of `h x w` output pixels under a square kernel.
    fn window(batch: usize, h: usize, w: usize, k: usize, s: usize, d: usize) -> ConvWindow {
        ConvWindow {
            batch,
            out_h: h,
            out_w: w,
            kernel: (k, k),
            stride: (s, s),
            dilation: (d, d),
        }
    }

    #[test]
    fn staging_needs_aligned_channel_groups() {
        let strides = [17 * 23 * 64, 23 * 64, 64];
        let w3 = window(2, 17, 23, 3, 1, 1);
        let t = ConvTile::Tile128x64;
        assert_eq!(
            select_conv_staging(64, 0x1000, strides, 0x2000, t, &w3),
            ConvStaging::Halo
        );
        // C_in not a multiple of 8.
        assert_eq!(
            select_conv_staging(3, 0x1000, [17 * 23 * 3, 23 * 3, 3], 0x2000, t, &w3),
            ConvStaging::Scalar
        );
        // Input view starting 3 channels into a buffer.
        assert_eq!(
            select_conv_staging(64, 0x1006, strides, 0x2000, t, &w3),
            ConvStaging::Scalar
        );
        // Weight view off 16-byte alignment.
        assert_eq!(
            select_conv_staging(64, 0x1000, strides, 0x2008, t, &w3),
            ConvStaging::Scalar
        );
        // Channel slice of an 84-channel buffer: pixel stride 84.
        assert_eq!(
            select_conv_staging(64, 0x1000, [17 * 23 * 84, 23 * 84, 84], 0x2000, t, &w3),
            ConvStaging::Scalar
        );
        // A size-1 dimension reports stride 0.
        assert_eq!(
            select_conv_staging(8, 0x1000, [0, 23 * 8, 8], 0x2000, t, &w3),
            ConvStaging::Halo
        );
    }

    #[test]
    fn halo_staging_needs_the_halo_to_fit() {
        let strides = [17 * 23 * 64, 23 * 64, 64];
        let pick = |tile: ConvTile, w: ConvWindow| select_conv_staging(64, 0, strides, 0, tile, &w);
        for tile in [
            ConvTile::Tile128x64,
            ConvTile::Tile256x32,
            ConvTile::Tile256x64,
        ] {
            // 3x3 stride 1: 10x18 = 180 <= 192 and 10x34 = 340 <= 352.
            assert_eq!(pick(tile, window(1, 9, 9, 3, 1, 1)), ConvStaging::Halo);
            // 1x1 stride 1: the patch itself.
            assert_eq!(pick(tile, window(1, 9, 9, 1, 1, 1)), ConvStaging::Halo);
            // Stride 2 doubles the halo; 5x5 and dilation 2 widen it.
            assert_eq!(pick(tile, window(1, 9, 9, 3, 2, 1)), ConvStaging::Vector);
            assert_eq!(pick(tile, window(1, 9, 9, 5, 1, 1)), ConvStaging::Vector);
            assert_eq!(pick(tile, window(1, 9, 9, 3, 1, 2)), ConvStaging::Vector);
        }
        assert_eq!(
            halo_pixels(ConvTile::Tile128x64, &window(1, 1, 1, 3, 1, 1)),
            Some(180)
        );
        assert_eq!(
            halo_pixels(ConvTile::Tile256x64, &window(1, 1, 1, 3, 1, 1)),
            Some(340)
        );
        assert_eq!(
            halo_pixels(ConvTile::Tile256x32, &window(1, 1, 1, 0, 1, 1)),
            None
        );
    }

    #[test]
    fn launch_config_puts_pixels_on_x() {
        let gather = ConvStaging::Vector;
        let w = window(1, 1080, 1920, 3, 1, 1);
        let cfg = conv_launch_config(&w, 64, ConvTile::Tile128x64, gather).unwrap();
        assert_eq!(cfg.grid_dim, (16_200, 1, 1));
        assert_eq!(cfg.block_dim, (128, 1, 1));
        let cfg = conv_launch_config(
            &window(2, 17, 23, 3, 1, 1),
            48,
            ConvTile::Tile128x64,
            gather,
        )
        .unwrap();
        assert_eq!(cfg.grid_dim, (7, 1, 1));
        let cfg = conv_launch_config(
            &window(1, 10, 10, 3, 1, 1),
            130,
            ConvTile::Tile128x64,
            gather,
        )
        .unwrap();
        assert_eq!(cfg.grid_dim, (1, 3, 1));
        let w4k = window(1, 2160, 3840, 3, 1, 1);
        let cfg = conv_launch_config(&w4k, 32, ConvTile::Tile256x32, gather).unwrap();
        assert_eq!(cfg.grid_dim, (32_400, 1, 1));
        let cfg = conv_launch_config(&w, 96, ConvTile::Tile256x64, gather).unwrap();
        assert_eq!(cfg.grid_dim, (8_100, 2, 1));
        assert_eq!(cfg.block_dim, (128, 1, 1));
    }

    #[test]
    fn halo_launch_config_counts_patches() {
        let halo = ConvStaging::Halo;
        // 1080 / 8 = 135 patch rows, 1920 / 16 = 120 patch columns.
        let w = window(1, 1080, 1920, 3, 1, 1);
        let cfg = conv_launch_config(&w, 64, ConvTile::Tile128x64, halo).unwrap();
        assert_eq!(cfg.grid_dim, (16_200, 1, 1));
        // 1920 / 32 = 60 patch columns.
        let cfg = conv_launch_config(&w, 96, ConvTile::Tile256x64, halo).unwrap();
        assert_eq!(cfg.grid_dim, (8_100, 2, 1));
        // 540 / 8 = 67.5 and 960 / 32 = 30: a partial last patch row.
        let cfg = conv_launch_config(
            &window(1, 540, 960, 3, 1, 1),
            32,
            ConvTile::Tile256x32,
            halo,
        )
        .unwrap();
        assert_eq!(cfg.grid_dim, (68 * 30, 1, 1));
        // Two images of 17x23: 3 patch rows, 1 patch column each.
        let cfg = conv_launch_config(&window(2, 17, 23, 3, 1, 1), 32, ConvTile::Tile256x32, halo)
            .unwrap();
        assert_eq!(cfg.grid_dim, (6, 1, 1));
        assert_eq!(cfg.block_dim, (128, 1, 1));
    }

    #[test]
    fn launch_config_rejects_empty_and_oversized_grids() {
        let gather = ConvStaging::Vector;
        let t = ConvTile::Tile128x64;
        assert!(conv_launch_config(&window(0, 4, 8, 3, 1, 1), 32, t, gather).is_none());
        assert!(conv_launch_config(&window(1, 4, 8, 3, 1, 1), 0, t, gather).is_none());
        assert!(conv_launch_config(&window(0, 4, 8, 3, 1, 1), 32, t, ConvStaging::Halo).is_none());
        let too_many_channels = (MAX_GRID_DIM_YZ as usize + 1) * 32;
        let w = window(1, 4, 8, 3, 1, 1);
        assert!(conv_launch_config(&w, too_many_channels, ConvTile::Tile256x32, gather).is_none());
    }

    #[test]
    fn kernel_names_match_the_cu_instantiations() {
        assert_eq!(
            conv_kernel_name(DType::F16, false, ConvTile::Tile256x32, ConvStaging::Scalar),
            "conv2d_implicit_gemm_f16_256x32_scalar"
        );
        assert_eq!(
            conv_kernel_name(
                DType::BF16,
                false,
                ConvTile::Tile128x64,
                ConvStaging::Scalar
            ),
            "conv2d_implicit_gemm_bf16_128x64_scalar"
        );
        assert_eq!(
            conv_kernel_name(DType::F16, false, ConvTile::Tile128x64, ConvStaging::Vector),
            "conv2d_implicit_gemm_f16_128x64_vec"
        );
        assert_eq!(
            conv_kernel_name(DType::F16, true, ConvTile::Tile256x32, ConvStaging::Vector),
            "conv2d_implicit_gemm_f16acc_256x32_vec"
        );
        assert_eq!(
            conv_kernel_name(DType::F16, true, ConvTile::Tile256x32, ConvStaging::Scalar),
            "conv2d_implicit_gemm_f16acc_256x32_scalar"
        );
        assert_eq!(
            conv_kernel_name(DType::F16, true, ConvTile::Tile256x64, ConvStaging::Vector),
            "conv2d_implicit_gemm_f16acc_256x64_vec"
        );
        assert_eq!(
            conv_kernel_name(DType::F16, true, ConvTile::Tile256x64, ConvStaging::Scalar),
            "conv2d_implicit_gemm_f16acc_256x64_scalar"
        );
        assert_eq!(
            conv_kernel_name(DType::F16, true, ConvTile::Tile256x64, ConvStaging::Halo),
            "conv2d_implicit_gemm_f16acc_256x64_halo"
        );
        assert_eq!(
            conv_kernel_name(DType::BF16, false, ConvTile::Tile128x64, ConvStaging::Halo),
            "conv2d_implicit_gemm_bf16_128x64_halo"
        );
    }
}
