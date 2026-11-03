//! CPU dispatch for the `ImageOps` gather ops.
//!
//! Strided input and output views pass straight to the kernel with their
//! strides, so no operand is copied to contiguous first.

use super::super::kernels::{GatherGeometry, GatherMap, image_gather_kernel};
use super::super::{CpuClient, CpuRuntime};
use crate::error::{Error, Result};
use crate::ops::common::{ImageMap, ImagePlan, NearestAxis, canonical_strides};
use crate::tensor::Tensor;

/// Source index table of one nearest axis.
fn nearest_table(axis: NearestAxis, in_size: usize, out_size: usize) -> Vec<usize> {
    (0..out_size)
        .map(|dst| axis.source_index(dst, in_size))
        .collect()
}

/// Run `plan` from `input` into `out`.
///
/// `out` already has the planned shape and the input's dtype, and it was
/// checked to address each element once and to share none with `input`.
pub fn image_run(
    client: &CpuClient,
    plan: &ImagePlan,
    input: &Tensor<CpuRuntime>,
    out: &Tensor<CpuRuntime>,
) -> Result<()> {
    if plan.numel() == 0 {
        return Ok(());
    }
    let geometry = GatherGeometry {
        out_dims: plan.out_dims,
        src_strides: canonical_strides(input.strides(), plan.layout),
        dst_strides: canonical_strides(out.strides(), plan.layout),
        channels_last: plan.channels_last(),
    };
    let rows: Vec<usize>;
    let cols: Vec<usize>;
    let map = match plan.map {
        ImageMap::Shuffle { factor } => GatherMap::Shuffle(factor),
        ImageMap::Unshuffle { factor } => GatherMap::Unshuffle(factor),
        ImageMap::Nearest { h, w } => {
            rows = nearest_table(h, plan.in_dims[2], plan.out_dims[2]);
            cols = nearest_table(w, plan.in_dims[3], plan.out_dims[3]);
            GatherMap::Nearest {
                rows: &rows,
                cols: &cols,
            }
        }
    };

    let src = input.ptr();
    let dst = out.ptr();
    // SAFETY: `ptr()` folds in each view's offset, the plan matches both
    // shapes, and `out` was checked against repeats and overlap.
    unsafe {
        match input.dtype().size_in_bytes() {
            1 => image_gather_kernel(client, src as *const u8, dst as *mut u8, &geometry, map),
            2 => image_gather_kernel(client, src as *const u16, dst as *mut u16, &geometry, map),
            4 => image_gather_kernel(client, src as *const u32, dst as *mut u32, &geometry, map),
            8 => image_gather_kernel(client, src as *const u64, dst as *mut u64, &geometry, map),
            16 => image_gather_kernel(
                client,
                src as *const [u64; 2],
                dst as *mut [u64; 2],
                &geometry,
                map,
            ),
            size => {
                return Err(Error::Internal(format!(
                    "{}: dtype {:?} has element size {size}, which no gather kernel moves",
                    plan.op,
                    input.dtype()
                )));
            }
        }
    }
    Ok(())
}
