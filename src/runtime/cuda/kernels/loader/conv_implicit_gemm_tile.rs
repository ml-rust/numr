//! Block-tile choice and launch geometry for the implicit-GEMM conv2d.
//!
//! `kernels/conv_implicit_gemm.cu` instantiates four block tiles and puts the
//! tile in the symbol name. The GEMM view is `M = N * H_out * W_out` rows by
//! `C_out` columns.

use crate::dtype::DType;
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;

use super::launch_dims::{LaunchConfig, MAX_GRID_DIM_X, MAX_GRID_DIM_YZ};
use super::matmul_wmma_tile::{
    WMMA_BLOCK_THREADS, WMMA_BLOCKS_PER_UNIT, WMMA_SMEM_BYTES, WmmaTile, select_wmma_tile,
};
use super::names::dtype_suffix;

/// Device waves the 128x64 grid must reach before it replaces 64x64 for
/// `32 < C_out <= 64`. Both tiles leave the same columns empty, so the
/// partial-edge rule of the GEMM selector applies.
const CONV_MIN_WAVES_128X64: u64 = 2;

/// Block tile of one implicit-GEMM conv kernel.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum ConvTile {
    /// 128x128, 4x4 warps, 2x2 fragments per warp.
    Tile128,
    /// 128x64, 4x4 warps, 2x1 fragments per warp.
    Tile128x64,
    /// 64x64, 4x4 warps, one fragment per warp.
    Tile64,
    /// 128x32, 8x2 warps, one fragment per warp.
    Tile128x32,
}

impl ConvTile {
    /// Output pixels one block computes.
    const fn tile_m(self) -> u32 {
        match self {
            Self::Tile128 | Self::Tile128x64 | Self::Tile128x32 => 128,
            Self::Tile64 => 64,
        }
    }

    /// Output channels one block computes.
    const fn tile_n(self) -> u32 {
        match self {
            Self::Tile128 => 128,
            Self::Tile128x64 | Self::Tile64 => 64,
            Self::Tile128x32 => 32,
        }
    }

    const fn suffix(self) -> &'static str {
        match self {
            Self::Tile128 => "128x128",
            Self::Tile128x64 => "128x64",
            Self::Tile64 => "64x64",
            Self::Tile128x32 => "128x32",
        }
    }
}

/// Symbol of one instantiation: `conv2d_implicit_gemm_{dtype}_{tile}`.
pub(super) fn conv_kernel_name(dtype: DType, tile: ConvTile) -> String {
    format!(
        "conv2d_implicit_gemm_{}_{}",
        dtype_suffix(dtype),
        tile.suffix()
    )
}

/// Pick the block tile for `m` output pixels and `c_out` channels.
///
/// The rule picks the tile that leaves the fewest output channels empty:
/// - `C_out <= 32`: 128x32. A 64-wide tile would leave at least half of its
///   columns empty.
/// - `32 < C_out <= 64`: one 64-wide column tile. 128x64 stages each B tile
///   once for twice the rows of 64x64, so it wins once its grid fills the
///   device [`CONV_MIN_WAVES_128X64`] times. A smaller grid takes 64x64 for
///   more blocks in flight.
/// - `C_out > 64`: the GEMM rule, [`select_wmma_tile`], on the `M x C_out`
///   output.
pub(super) fn select_conv_tile(m: usize, c_out: usize, k: usize, device_index: usize) -> ConvTile {
    let units = CudaDevice::new(device_index).profile().compute_units;
    if let Some(tile) = narrow_conv_tile(m, c_out, units) {
        return tile;
    }
    match select_wmma_tile(m, c_out, k, 1, device_index) {
        WmmaTile::Tile128 => ConvTile::Tile128,
        WmmaTile::Tile128x64 => ConvTile::Tile128x64,
        WmmaTile::Tile64 => ConvTile::Tile64,
    }
}

/// The tile for `C_out <= 64`, or `None` when the GEMM rule decides.
fn narrow_conv_tile(m: usize, c_out: usize, compute_units: u32) -> Option<ConvTile> {
    if c_out <= 32 {
        return Some(ConvTile::Tile128x32);
    }
    if c_out > 64 {
        return None;
    }
    let wave = u64::from(compute_units) * u64::from(WMMA_BLOCKS_PER_UNIT);
    let blocks = (m as u64).div_ceil(u64::from(ConvTile::Tile128x64.tile_m()));
    if blocks >= wave * CONV_MIN_WAVES_128X64 {
        Some(ConvTile::Tile128x64)
    } else {
        Some(ConvTile::Tile64)
    }
}

/// Grid and block for one launch: pixels on grid x, channels on grid y.
///
/// Pixels go on x because a large frame can need more than 65535 row tiles.
/// Returns `None` when either extent passes the CUDA grid limit.
pub(super) fn conv_launch_config(m: usize, c_out: usize, tile: ConvTile) -> Option<LaunchConfig> {
    let grid_x = (m as u64).div_ceil(u64::from(tile.tile_m()));
    let grid_y = (c_out as u64).div_ceil(u64::from(tile.tile_n()));
    if grid_x == 0 || grid_y == 0 {
        return None;
    }
    if grid_x > u64::from(MAX_GRID_DIM_X) || grid_y > u64::from(MAX_GRID_DIM_YZ) {
        return None;
    }
    Some(LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        // Equals `CONV_IGEMM_THREADS` and uses static shared memory only,
        // as the WMMA GEMM does.
        block_dim: (WMMA_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: WMMA_SMEM_BYTES,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNITS: u32 = 28;

    #[test]
    fn narrow_channels_take_128x32() {
        for c_out in [1, 3, 16, 32] {
            assert_eq!(
                narrow_conv_tile(1 << 20, c_out, UNITS),
                Some(ConvTile::Tile128x32)
            );
        }
    }

    #[test]
    fn mid_channels_take_a_64_wide_tile() {
        assert_eq!(
            narrow_conv_tile(1 << 20, 48, UNITS),
            Some(ConvTile::Tile128x64)
        );
        assert_eq!(
            narrow_conv_tile(1 << 20, 64, UNITS),
            Some(ConvTile::Tile128x64)
        );
        // 8 row tiles cannot fill 2 waves of 56 blocks.
        assert_eq!(narrow_conv_tile(1024, 64, UNITS), Some(ConvTile::Tile64));
    }

    #[test]
    fn wide_channels_defer_to_the_gemm_rule() {
        assert_eq!(narrow_conv_tile(1 << 20, 65, UNITS), None);
        assert_eq!(narrow_conv_tile(1 << 20, 256, UNITS), None);
    }

    #[test]
    fn launch_config_puts_pixels_on_x() {
        let cfg = conv_launch_config(3840 * 2160, 32, ConvTile::Tile128x32).unwrap();
        assert_eq!(cfg.grid_dim, (64_800, 1, 1));
        assert_eq!(cfg.block_dim, (512, 1, 1));
        let cfg = conv_launch_config(17 * 23 * 2, 48, ConvTile::Tile64).unwrap();
        assert_eq!(cfg.grid_dim, (13, 1, 1));
        let cfg = conv_launch_config(100, 130, ConvTile::Tile128).unwrap();
        assert_eq!(cfg.grid_dim, (1, 2, 1));
    }

    #[test]
    fn launch_config_rejects_empty_and_oversized_grids() {
        assert!(conv_launch_config(0, 32, ConvTile::Tile64).is_none());
        assert!(conv_launch_config(32, 0, ConvTile::Tile64).is_none());
        let too_many_channels = (MAX_GRID_DIM_YZ as usize + 1) * 32;
        assert!(conv_launch_config(32, too_many_channels, ConvTile::Tile128x32).is_none());
    }

    #[test]
    fn kernel_names_match_the_cu_instantiations() {
        assert_eq!(
            conv_kernel_name(DType::F16, ConvTile::Tile128x32),
            "conv2d_implicit_gemm_f16_128x32"
        );
        assert_eq!(
            conv_kernel_name(DType::BF16, ConvTile::Tile128),
            "conv2d_implicit_gemm_bf16_128x128"
        );
    }
}
