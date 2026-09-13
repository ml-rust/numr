//! Row-wise pad launcher: padding confined to the last two dimensions.
//!
//! `launch_pad` in `shape.rs` routes here whenever every leading dimension
//! keeps its extent. The tensor is treated as `batch` matrices; the kernel
//! walks columns across consecutive threads, so every access coalesces.
//! When both base pointers and every row width are 16-byte multiples the
//! copy runs one 128-bit chunk per thread instead of one element.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream, LaunchArgs};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, MAX_GRID_DIM_X, MAX_GRID_DIM_YZ, get_kernel_function, get_or_load_module,
    kernel_name, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Module name for the row-wise pad kernels.
pub const PAD_ROWS_MODULE: &str = "pad_rows";

/// Width of one vectorised chunk in bytes.
const VEC_BYTES: usize = 16;

/// Columns each scalar-kernel thread owns. Must match `PAD_ROWS_ITEMS` in
/// `pad_rows.cu`.
const SCALAR_ITEMS: usize = 4;

/// Every dtype rendering of a pad fill value, held together so a kernel
/// argument can borrow the one matching the launch dtype.
pub struct PadFill {
    f32: f32,
    f64: f64,
    i32: i32,
    i64: i64,
    u32: u32,
    u64: u64,
    i16: i16,
    i8: i8,
    u16: u16,
    u8: u8,
    #[cfg(feature = "f16")]
    f16: half::f16,
    #[cfg(feature = "f16")]
    bf16: half::bf16,
    #[cfg(feature = "fp8")]
    fp8_e4m3: crate::dtype::FP8E4M3,
    #[cfg(feature = "fp8")]
    fp8_e5m2: crate::dtype::FP8E5M2,
}

impl PadFill {
    pub fn new(value: f64) -> Self {
        Self {
            f32: value as f32,
            f64: value,
            i32: value as i32,
            i64: value as i64,
            u32: value as u32,
            u64: value as u64,
            i16: value as i16,
            i8: value as i8,
            u16: value as u16,
            u8: value as u8,
            #[cfg(feature = "f16")]
            f16: half::f16::from_f64(value),
            #[cfg(feature = "f16")]
            bf16: half::bf16::from_f64(value),
            #[cfg(feature = "fp8")]
            fp8_e4m3: crate::dtype::FP8E4M3::from_f32(value as f32),
            #[cfg(feature = "fp8")]
            fp8_e5m2: crate::dtype::FP8E5M2::from_f32(value as f32),
        }
    }

    /// Push the fill value for `dtype` as the next kernel argument.
    pub fn push_arg<'a>(&'a self, builder: &mut LaunchArgs<'a>, dtype: DType) -> Result<()> {
        match dtype {
            DType::F32 => builder.arg(&self.f32),
            DType::F64 => builder.arg(&self.f64),
            DType::I32 => builder.arg(&self.i32),
            DType::I64 => builder.arg(&self.i64),
            DType::U32 => builder.arg(&self.u32),
            DType::U64 => builder.arg(&self.u64),
            DType::I16 => builder.arg(&self.i16),
            DType::I8 => builder.arg(&self.i8),
            DType::U16 => builder.arg(&self.u16),
            DType::U8 => builder.arg(&self.u8),
            #[cfg(feature = "f16")]
            DType::F16 => builder.arg(&self.f16),
            #[cfg(feature = "f16")]
            DType::BF16 => builder.arg(&self.bf16),
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => builder.arg(&self.fp8_e4m3),
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => builder.arg(&self.fp8_e5m2),
            _ => return Err(Error::UnsupportedDType { dtype, op: "pad" }),
        };
        Ok(())
    }
}

/// Geometry of a row-wise pad: `batch` matrices of `src_rows x src_cols`
/// padded to `out_rows x out_cols`. Rank-1 tensors use `src_rows == 1`.
#[derive(Clone, Copy, Debug)]
pub struct PadRowsGeometry {
    pub batch: usize,
    pub src_rows: usize,
    pub src_cols: usize,
    pub out_rows: usize,
    pub out_cols: usize,
    pub pad_before_row: usize,
    pub pad_before_col: usize,
}

impl PadRowsGeometry {
    /// Fold a pad request into row geometry when every dimension before the
    /// last two keeps its extent. Returns `None` when a leading dimension is
    /// padded, which sends the launch to the generic kernel.
    pub fn from_pad(
        src_shape: &[usize],
        out_shape: &[usize],
        pad_before: &[usize],
    ) -> Option<Self> {
        let ndim = src_shape.len();
        if ndim == 0 {
            return None;
        }
        let lead = ndim.saturating_sub(2);
        let leading_untouched =
            (0..lead).all(|d| src_shape[d] == out_shape[d] && pad_before[d] == 0);
        if !leading_untouched {
            return None;
        }
        let batch: usize = src_shape[..lead].iter().product();
        let (src_rows, out_rows, pad_before_row) = if ndim >= 2 {
            (
                src_shape[ndim - 2],
                out_shape[ndim - 2],
                pad_before[ndim - 2],
            )
        } else {
            (1, 1, 0)
        };
        Some(Self {
            batch,
            src_rows,
            src_cols: src_shape[ndim - 1],
            out_rows,
            out_cols: out_shape[ndim - 1],
            pad_before_row,
            pad_before_col: pad_before[ndim - 1],
        })
    }

    /// True when every row on both sides starts on a 16-byte boundary, so
    /// the copy can move whole 128-bit chunks. The three width checks are the
    /// row-stride half of the guard; the pointer checks are the base half.
    fn vector_ok(&self, src_ptr: u64, dst_ptr: u64, elem: usize) -> bool {
        let aligned = |bytes: usize| bytes.is_multiple_of(VEC_BYTES);
        src_ptr.is_multiple_of(VEC_BYTES as u64)
            && dst_ptr.is_multiple_of(VEC_BYTES as u64)
            && aligned(self.src_cols * elem)
            && aligned(self.out_cols * elem)
            && aligned(self.pad_before_col * elem)
    }
}

/// Launch the row-wise pad kernel.
///
/// # Safety
///
/// - `src_ptr` points to contiguous device memory holding
///   `batch * src_rows * src_cols` elements of `dtype`
/// - `dst_ptr` points to device memory for `batch * out_rows * out_cols`
///   elements of `dtype`
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_pad_rows(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    fill: &PadFill,
    geom: PadRowsGeometry,
) -> Result<()> {
    let elem = dtype.size_in_bytes();
    let vectorised = geom.vector_ok(src_ptr, dst_ptr, elem);
    let lanes = if vectorised { VEC_BYTES / elem } else { 1 };
    let (base, src_cols, out_cols, pad_before_col) = if vectorised {
        (
            "pad_rows_vec",
            geom.src_cols / lanes,
            geom.out_cols / lanes,
            geom.pad_before_col / lanes,
        )
    } else {
        (
            "pad_rows",
            geom.src_cols,
            geom.out_cols,
            geom.pad_before_col,
        )
    };

    let cols_per_block = BLOCK_SIZE as usize * if vectorised { 1 } else { SCALAR_ITEMS };
    let grid_x = out_cols.div_ceil(cols_per_block).max(1);
    if grid_x > MAX_GRID_DIM_X as usize {
        return Err(Error::InvalidArgument {
            arg: "out_cols",
            reason: format!(
                "{} output columns need {grid_x} blocks, exceeding the CUDA max grid \
                 extent of {MAX_GRID_DIM_X}",
                geom.out_cols
            ),
        });
    }
    let grid_y = geom.out_rows.min(MAX_GRID_DIM_YZ as usize).max(1) as u32;
    let grid_z = geom.batch.min(MAX_GRID_DIM_YZ as usize).max(1) as u32;

    unsafe {
        let module = get_or_load_module(context, device_index, PAD_ROWS_MODULE)?;
        let func_name = kernel_name(base, dtype);
        let func = get_kernel_function(&module, &func_name)?;
        let cfg = launch_config((grid_x as u32, grid_y, grid_z), (BLOCK_SIZE, 1, 1), 0);

        let batch = geom.batch as u32;
        let src_rows = geom.src_rows as u32;
        let src_cols = src_cols as u32;
        let out_rows = geom.out_rows as u32;
        let out_cols = out_cols as u32;
        let pad_before_row = geom.pad_before_row as u32;
        let pad_before_col = pad_before_col as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        fill.push_arg(&mut builder, dtype)?;
        builder.arg(&batch);
        builder.arg(&src_rows);
        builder.arg(&src_cols);
        builder.arg(&out_rows);
        builder.arg(&out_cols);
        builder.arg(&pad_before_row);
        builder.arg(&pad_before_col);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA {func_name} kernel launch failed: {e:?}"))
        })?;
    }
    Ok(())
}
