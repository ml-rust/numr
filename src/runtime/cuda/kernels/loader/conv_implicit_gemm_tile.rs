//! Accumulator, block tile and launch geometry for the implicit-GEMM
//! conv2d.
//!
//! `kernels/conv_implicit_gemm.cu` instantiates each accumulator with block
//! tiles, each with five staging variants, and names the symbol after all
//! three. The staging variant comes from
//! [`super::conv_implicit_gemm_staging`]. The GEMM view is
//! `M = N * H_out * W_out` rows by `C_out` columns.

use crate::dtype::DType;

use super::conv_implicit_gemm_staging::ConvStaging;
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
    pub(super) const fn patch(self) -> (usize, usize) {
        match self {
            Self::Tile128x64 => (8, 16),
            Self::Tile256x32 | Self::Tile256x64 => (8, 32),
        }
    }

    /// Most halo pixels one halo-staging block holds. Equals
    /// `ConvTileCfg::HALO_PIX` of the instantiation.
    pub(super) const fn halo_capacity(self) -> usize {
        match self {
            Self::Tile128x64 => 192,
            Self::Tile256x32 | Self::Tile256x64 => 352,
        }
    }

    /// Most 4-pixel runs one NCHW halo-staging block stages per chunk.
    /// Equals `ConvTileCfg::NCHW_RUNS` of the instantiation.
    pub(super) const fn nchw_run_capacity(self) -> usize {
        match self {
            Self::Tile128x64 => 64,
            Self::Tile256x32 | Self::Tile256x64 => 104,
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
    /// Left padding. It sets the input column of each NCHW halo run.
    pub pad_left: usize,
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
/// Grid x counts `tile_m`-row tiles of `M = N * H_out * W_out`, or for a
/// halo staging ([`ConvStaging::is_halo`]) the output patches of every
/// image, in `(n, patch row, patch column)` order. Pixels go on x because a large
/// frame can need more than 65535 blocks. Returns `None` when either
/// extent passes the CUDA grid limit or is 0.
pub(super) fn conv_launch_config(
    window: &ConvWindow,
    c_out: usize,
    tile: ConvTile,
    staging: ConvStaging,
) -> Option<LaunchConfig> {
    let grid_x = if staging.is_halo() {
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

/// `batch` images of `h x w` output pixels under a square kernel, with
/// left padding `k / 2`.
#[cfg(test)]
pub(super) fn test_window(
    batch: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    d: usize,
) -> ConvWindow {
    ConvWindow {
        batch,
        out_h: h,
        out_w: w,
        kernel: (k, k),
        stride: (s, s),
        dilation: (d, d),
        pad_left: k / 2,
    }
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

    use super::test_window as window;

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
        assert_eq!(
            conv_kernel_name(
                DType::F16,
                false,
                ConvTile::Tile128x64,
                ConvStaging::NchwHalo
            ),
            "conv2d_implicit_gemm_f16_128x64_nchw_halo"
        );
        assert_eq!(
            conv_kernel_name(
                DType::BF16,
                false,
                ConvTile::Tile256x32,
                ConvStaging::NchwGather
            ),
            "conv2d_implicit_gemm_bf16_256x32_nchw_gather"
        );
        assert_eq!(
            conv_kernel_name(
                DType::F16,
                true,
                ConvTile::Tile256x64,
                ConvStaging::NchwHalo
            ),
            "conv2d_implicit_gemm_f16acc_256x64_nchw_halo"
        );
        assert_eq!(
            conv_kernel_name(
                DType::F16,
                true,
                ConvTile::Tile256x32,
                ConvStaging::NchwGather
            ),
            "conv2d_implicit_gemm_f16acc_256x32_nchw_gather"
        );
    }

    #[test]
    fn nchw_halo_launch_config_counts_patches() {
        let w = window(1, 1080, 1920, 3, 1, 1);
        let cfg = conv_launch_config(&w, 64, ConvTile::Tile128x64, ConvStaging::NchwHalo).unwrap();
        assert_eq!(cfg.grid_dim, (16_200, 1, 1));
        let cfg =
            conv_launch_config(&w, 64, ConvTile::Tile128x64, ConvStaging::NchwGather).unwrap();
        assert_eq!(cfg.grid_dim, (16_200, 1, 1));
        let cfg = conv_launch_config(
            &window(2, 17, 23, 3, 1, 1),
            32,
            ConvTile::Tile256x32,
            ConvStaging::NchwHalo,
        )
        .unwrap();
        assert_eq!(cfg.grid_dim, (6, 1, 1));
    }
}
