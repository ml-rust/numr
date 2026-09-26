//! Reflect-mode pad: mirrors the interior of each padded dimension into the
//! padded region, excluding the edge element (PyTorch `F.pad(mode="reflect")`
//! semantics).

use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::common::shape_ops;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Pad tensor by reflecting its interior, excluding the edge element.
pub fn pad_reflect_impl(
    client: &CpuClient,
    tensor: &Tensor<CpuRuntime>,
    padding: &[usize],
) -> Result<Tensor<CpuRuntime>> {
    // Use shared validation
    let params = shape_ops::validate_reflect_pad(tensor, padding)?;

    // Handle case where no padding is added
    if params.pad_per_dim.iter().all(|&(b, a)| b == 0 && a == 0) {
        return Ok(tensor.clone());
    }

    let dtype = tensor.dtype();
    let in_shape = tensor.shape();

    let out = Tensor::<CpuRuntime>::empty(&params.out_shape, dtype, &client.device)?;

    // Make input contiguous
    let tensor_contig = ensure_contiguous(tensor)?;
    let src_ptr = tensor_contig.ptr();
    let dst_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            pad_reflect_copy_kernel::<T>(
                src_ptr as *const T,
                dst_ptr as *mut T,
                in_shape,
                &params.out_shape,
                &params.pad_per_dim,
            );
        }
    }, "pad_reflect");

    Ok(out)
}

/// Map one output coordinate back to its source coordinate under reflect
/// padding. `before` and `size` come from the same dimension; `validate_reflect_pad`
/// guarantees `before < size` and `after < size`, so a single mirror bounce
/// (never a double reflection) always lands back inside `0..size`.
#[inline]
fn reflect_coord(out_coord: usize, before: usize, size: usize) -> usize {
    let rel = out_coord as isize - before as isize;
    if rel < 0 {
        (-rel) as usize
    } else if (rel as usize) < size {
        rel as usize
    } else {
        let over = rel - size as isize;
        (size as isize - 2 - over) as usize
    }
}

/// Kernel for copying input data into a reflect-padded output
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pad_reflect_copy_kernel<T: Element>(
    src: *const T,
    dst: *mut T,
    in_shape: &[usize],
    out_shape: &[usize],
    pad_per_dim: &[(usize, usize)],
) {
    let ndim = in_shape.len();
    let out_numel: usize = out_shape.iter().product();

    // Compute strides for input and output
    let mut in_strides = vec![1usize; ndim];
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // For each output element, gather from its reflected source coordinate
    for out_idx in 0..out_numel {
        let mut remaining = out_idx;
        let mut src_idx = 0usize;

        #[allow(clippy::needless_range_loop)]
        for d in 0..ndim {
            let out_coord = remaining / out_strides[d];
            remaining %= out_strides[d];

            let (before, _after) = pad_per_dim[d];
            let src_coord = reflect_coord(out_coord, before, in_shape[d]);
            src_idx += src_coord * in_strides[d];
        }

        *dst.add(out_idx) = *src.add(src_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{PadMode, ShapeOps};
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    fn client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    /// PyTorch: `F.pad(torch.tensor([1.,2.,3.,4.,5.]), (2, 2), mode="reflect")`
    /// -> `[3, 2, 1, 2, 3, 4, 5, 4, 3]`.
    #[test]
    fn reflect_pad_1d_matches_pytorch() {
        let (client, device) = client();
        let x =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device).unwrap();
        let out = client.pad_mode(&x, &[2, 2], PadMode::Reflect).unwrap();
        assert_eq!(out.shape(), &[9]);
        let v: Vec<f32> = out.to_vec();
        assert_eq!(v, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    /// PyTorch: `F.pad(x, (1, 1), mode="reflect")` on a `[2, 2, 3]` tensor
    /// pads only the last (T) axis, independently per (B, C) row.
    /// Row (0, 0) = `[1, 2, 3]` -> `[2, 1, 2, 3, 2]`.
    /// Row (1, 1) = `[10, 11, 12]` -> `[11, 10, 11, 12, 11]`.
    #[test]
    fn reflect_pad_last_dim_of_b_c_t_matches_pytorch() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 2.0, 3.0, // (0, 0)
                4.0, 5.0, 6.0, // (0, 1)
                7.0, 8.0, 9.0, // (1, 0)
                10.0, 11.0, 12.0, // (1, 1)
            ],
            &[2, 2, 3],
            &device,
        )
        .unwrap();
        let out = client.pad_mode(&x, &[1, 1], PadMode::Reflect).unwrap();
        assert_eq!(out.shape(), &[2, 2, 5]);
        let v: Vec<f32> = out.to_vec();
        assert_eq!(&v[0..5], &[2.0, 1.0, 2.0, 3.0, 2.0]);
        assert_eq!(&v[15..20], &[11.0, 10.0, 11.0, 12.0, 11.0]);
    }

    /// PyTorch: `F.pad(x, (1, 1, 1, 1), mode="reflect")` on the 3x3 tensor
    /// `[[1,2,3],[4,5,6],[7,8,9]]` reflects both trailing axes:
    /// ```text
    /// [[5,4,5,6,5],
    ///  [2,1,2,3,2],
    ///  [5,4,5,6,5],
    ///  [8,7,8,9,8],
    ///  [5,4,5,6,5]]
    /// ```
    #[test]
    fn reflect_pad_two_dims_matches_pytorch() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            &device,
        )
        .unwrap();
        let out = client
            .pad_mode(&x, &[1, 1, 1, 1], PadMode::Reflect)
            .unwrap();
        assert_eq!(out.shape(), &[5, 5]);
        let v: Vec<f32> = out.to_vec();
        assert_eq!(
            v,
            vec![
                5.0, 4.0, 5.0, 6.0, 5.0, //
                2.0, 1.0, 2.0, 3.0, 2.0, //
                5.0, 4.0, 5.0, 6.0, 5.0, //
                8.0, 7.0, 8.0, 9.0, 8.0, //
                5.0, 4.0, 5.0, 6.0, 5.0,
            ]
        );
    }

    /// A pad size equal to (or larger than) the dimension's size has no
    /// defined source element to reflect from, and must error rather than
    /// read out of bounds.
    #[test]
    fn reflect_pad_rejects_pad_at_least_dim_size() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        assert!(client.pad_mode(&x, &[2, 0], PadMode::Reflect).is_err());
        assert!(client.pad_mode(&x, &[0, 2], PadMode::Reflect).is_err());
        assert!(client.pad_mode(&x, &[3, 0], PadMode::Reflect).is_err());
    }
}
