//! Shared helpers for reduction operations

/// Use a fused single-pass path for small multi-dimension contiguous reductions.
///
/// This avoids intermediate tensor allocations from sequential dim-by-dim reduction.
/// The 1 MiB threshold is chosen to fit comfortably within L1/L2 cache on modern CPUs,
/// ensuring the single-pass scan has good cache locality.
pub(super) const FUSED_MULTI_DIM_REDUCTION_MAX_BYTES: usize = 1 << 20; // 1 MiB

use crate::runtime::cpu::CpuRuntime;
use crate::tensor::Tensor;

#[inline]
pub(super) fn should_fuse_multi_dim_reduction(a: &Tensor<CpuRuntime>, dims: &[usize]) -> bool {
    if dims.len() <= 1 || !a.is_contiguous() {
        return false;
    }

    if a.numel() == 0 {
        return false;
    }

    let bytes = a.numel().saturating_mul(a.dtype().size_in_bytes());
    bytes <= FUSED_MULTI_DIM_REDUCTION_MAX_BYTES
}

#[inline]
pub(super) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[inline]
pub(super) fn out_index_from_coord(
    coord: &[usize],
    reduce_mask: &[bool],
    keepdim: bool,
    kept_axes: &[usize],
    out_strides: &[usize],
) -> usize {
    if out_strides.is_empty() {
        return 0;
    }

    if keepdim {
        let mut idx = 0usize;
        for axis in 0..coord.len() {
            if !reduce_mask[axis] {
                idx += coord[axis] * out_strides[axis];
            }
        }
        idx
    } else {
        let mut idx = 0usize;
        for (out_axis, &axis) in kept_axes.iter().enumerate() {
            idx += coord[axis] * out_strides[out_axis];
        }
        idx
    }
}

#[inline]
pub(super) fn advance_coord(coord: &mut [usize], shape: &[usize]) {
    for axis in (0..coord.len()).rev() {
        coord[axis] += 1;
        if coord[axis] < shape[axis] {
            return;
        }
        coord[axis] = 0;
    }
}
