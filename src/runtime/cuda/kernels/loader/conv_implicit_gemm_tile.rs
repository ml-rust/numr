//! Block tile, staging variant and launch geometry for the implicit-GEMM
//! conv2d.
//!
//! `kernels/conv_implicit_gemm.cu` instantiates two block tiles, each with
//! two staging variants, and names the symbol after both. The GEMM view is
//! `M = N * H_out * W_out` rows by `C_out` columns.

use crate::dtype::DType;

use super::launch_dims::{LaunchConfig, MAX_GRID_DIM_X, MAX_GRID_DIM_YZ};
use super::names::dtype_suffix;

/// Block tile of one implicit-GEMM conv kernel. Every tile has a 64x32
/// warp tile.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum ConvTile {
    /// 128x64, 4 warps on a 2x2 grid.
    Tile128x64,
    /// 256x32, 4 warps on a 4x1 grid, for `C_out <= 32`.
    Tile256x32,
}

impl ConvTile {
    /// Output pixels one block computes.
    const fn tile_m(self) -> u32 {
        match self {
            Self::Tile256x32 => 256,
            Self::Tile128x64 => 128,
        }
    }

    /// Output channels one block computes.
    const fn tile_n(self) -> u32 {
        match self {
            Self::Tile128x64 => 64,
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
        }
    }
}

/// Operand staging variant of one implicit-GEMM conv kernel.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum ConvStaging {
    /// 16-byte loads, K in tap-major order.
    Vector,
    /// 16-bit gathers, K in flat `(kh, kw, ci)` order.
    Scalar,
}

impl ConvStaging {
    const fn suffix(self) -> &'static str {
        match self {
            Self::Vector => "vec",
            Self::Scalar => "scalar",
        }
    }
}

/// Symbol of one instantiation: `conv2d_implicit_gemm_{dtype}_{tile}_{staging}`.
pub(super) fn conv_kernel_name(dtype: DType, tile: ConvTile, staging: ConvStaging) -> String {
    format!(
        "conv2d_implicit_gemm_{}_{}_{}",
        dtype_suffix(dtype),
        tile.suffix(),
        staging.suffix()
    )
}

/// Pick the staging variant.
///
/// The vector variant reads 8 channels of one tap per 16-byte load. That
/// needs `C_in % 8 == 0`, a 16-byte aligned input and weight, and input
/// N/H/W element strides that are multiples of 8. Strides are never
/// negative here; `ops/cuda/conv_fused_native.rs` refuses those views.
pub(super) fn select_conv_staging(
    c_in: usize,
    input_ptr: u64,
    input_strides: [i64; 3],
    weight_ptr: u64,
) -> ConvStaging {
    let aligned = |ptr: u64| ptr.is_multiple_of(16);
    if c_in.is_multiple_of(8)
        && aligned(input_ptr)
        && aligned(weight_ptr)
        && input_strides.iter().all(|s| s % 8 == 0)
    {
        ConvStaging::Vector
    } else {
        ConvStaging::Scalar
    }
}

/// Pick the block tile for `c_out` channels.
///
/// - `C_out <= 32`: 256x32. A 64-wide tile leaves half of its columns empty.
/// - Otherwise 128x64. Channels past 64 go over grid y.
pub(super) fn select_conv_tile(c_out: usize) -> ConvTile {
    if c_out <= 32 {
        ConvTile::Tile256x32
    } else {
        ConvTile::Tile128x64
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
            assert_eq!(select_conv_tile(c_out), ConvTile::Tile256x32);
        }
    }

    #[test]
    fn wide_channels_take_128x64() {
        for c_out in [33, 48, 64, 96, 160] {
            assert_eq!(select_conv_tile(c_out), ConvTile::Tile128x64);
        }
    }

    #[test]
    fn staging_needs_aligned_channel_groups() {
        let strides = [17 * 23 * 64, 23 * 64, 64];
        assert_eq!(
            select_conv_staging(64, 0x1000, strides, 0x2000),
            ConvStaging::Vector
        );
        // C_in not a multiple of 8.
        assert_eq!(
            select_conv_staging(3, 0x1000, [17 * 23 * 3, 23 * 3, 3], 0x2000),
            ConvStaging::Scalar
        );
        // Input view starting 3 channels into a buffer.
        assert_eq!(
            select_conv_staging(64, 0x1006, strides, 0x2000),
            ConvStaging::Scalar
        );
        // Weight view off 16-byte alignment.
        assert_eq!(
            select_conv_staging(64, 0x1000, strides, 0x2008),
            ConvStaging::Scalar
        );
        // Channel slice of an 84-channel buffer: pixel stride 84.
        assert_eq!(
            select_conv_staging(64, 0x1000, [17 * 23 * 84, 23 * 84, 84], 0x2000),
            ConvStaging::Scalar
        );
        // A size-1 dimension reports stride 0.
        assert_eq!(
            select_conv_staging(8, 0x1000, [0, 23 * 8, 8], 0x2000),
            ConvStaging::Vector
        );
    }

    #[test]
    fn launch_config_puts_pixels_on_x() {
        let cfg = conv_launch_config(1080 * 1920, 64, ConvTile::Tile128x64).unwrap();
        assert_eq!(cfg.grid_dim, (16_200, 1, 1));
        assert_eq!(cfg.block_dim, (128, 1, 1));
        let cfg = conv_launch_config(17 * 23 * 2, 48, ConvTile::Tile128x64).unwrap();
        assert_eq!(cfg.grid_dim, (7, 1, 1));
        assert_eq!(cfg.block_dim, (128, 1, 1));
        let cfg = conv_launch_config(100, 130, ConvTile::Tile128x64).unwrap();
        assert_eq!(cfg.grid_dim, (1, 3, 1));
        let cfg = conv_launch_config(3840 * 2160, 32, ConvTile::Tile256x32).unwrap();
        assert_eq!(cfg.grid_dim, (32_400, 1, 1));
        assert_eq!(cfg.block_dim, (128, 1, 1));
    }

    #[test]
    fn launch_config_rejects_empty_and_oversized_grids() {
        assert!(conv_launch_config(0, 32, ConvTile::Tile128x64).is_none());
        assert!(conv_launch_config(32, 0, ConvTile::Tile128x64).is_none());
        let too_many_channels = (MAX_GRID_DIM_YZ as usize + 1) * 32;
        assert!(conv_launch_config(32, too_many_channels, ConvTile::Tile256x32).is_none());
    }

    #[test]
    fn kernel_names_match_the_cu_instantiations() {
        assert_eq!(
            conv_kernel_name(DType::F16, ConvTile::Tile256x32, ConvStaging::Scalar),
            "conv2d_implicit_gemm_f16_256x32_scalar"
        );
        assert_eq!(
            conv_kernel_name(DType::BF16, ConvTile::Tile128x64, ConvStaging::Scalar),
            "conv2d_implicit_gemm_bf16_128x64_scalar"
        );
        assert_eq!(
            conv_kernel_name(DType::F16, ConvTile::Tile128x64, ConvStaging::Vector),
            "conv2d_implicit_gemm_f16_128x64_vec"
        );
    }
}
