//! Replicate-mode pad: every out-of-range index clamps to the nearest edge
//! element (PyTorch `F.pad(mode="replicate")` semantics).

use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::common::pad_replicate::validate_replicate_pad;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Pad tensor by repeating its edge elements.
pub fn pad_replicate_impl(
    client: &CpuClient,
    tensor: &Tensor<CpuRuntime>,
    padding: &[usize],
) -> Result<Tensor<CpuRuntime>> {
    let params = validate_replicate_pad(tensor, padding)?;

    if params.pad_per_dim.iter().all(|&(b, a)| b == 0 && a == 0) {
        return Ok(tensor.clone());
    }

    let dtype = tensor.dtype();
    let in_shape = tensor.shape();
    let out = Tensor::<CpuRuntime>::empty(&params.out_shape, dtype, &client.device)?;

    let tensor_contig = ensure_contiguous(tensor)?;
    let src_ptr = tensor_contig.ptr();
    let dst_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            pad_replicate_copy_kernel::<T>(
                src_ptr as *const T,
                dst_ptr as *mut T,
                in_shape,
                &params.out_shape,
                &params.pad_per_dim,
            );
        }
    }, "pad_replicate");

    Ok(out)
}

/// Map one output coordinate to its source coordinate: `clamp(out - before, 0, size - 1)`.
/// `validate_replicate_pad` guarantees `size >= 1` whenever the output is non-empty.
#[inline]
fn replicate_coord(out_coord: usize, before: usize, size: usize) -> usize {
    out_coord.saturating_sub(before).min(size - 1)
}

/// Copy input data into a replicate-padded output.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pad_replicate_copy_kernel<T: Element>(
    src: *const T,
    dst: *mut T,
    in_shape: &[usize],
    out_shape: &[usize],
    pad_per_dim: &[(usize, usize)],
) {
    let ndim = in_shape.len();
    let out_numel: usize = out_shape.iter().product();

    let mut in_strides = vec![1usize; ndim];
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    for out_idx in 0..out_numel {
        let mut remaining = out_idx;
        let mut src_idx = 0usize;

        #[allow(clippy::needless_range_loop)]
        for d in 0..ndim {
            let out_coord = remaining / out_strides[d];
            remaining %= out_strides[d];

            let (before, _after) = pad_per_dim[d];
            src_idx += replicate_coord(out_coord, before, in_shape[d]) * in_strides[d];
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

    /// PyTorch: `F.pad(torch.tensor([1.,2.,3.]), (2, 1), mode="replicate")`
    /// -> `[1, 1, 1, 2, 3, 3]`.
    #[test]
    fn replicate_pad_1d_matches_pytorch() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let out = client.pad_mode(&x, &[2, 1], PadMode::Replicate).unwrap();
        assert_eq!(out.shape(), &[6]);
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
    }

    /// Pad larger than the dimension: `[1, 2]` padded by (5, 5) repeats each edge.
    #[test]
    fn replicate_pad_larger_than_input() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let out = client.pad_mode(&x, &[5, 5], PadMode::Replicate).unwrap();
        let mut want = vec![1.0f32; 6];
        want.extend([2.0f32; 6]);
        assert_eq!(out.to_vec::<f32>(), want);
    }

    #[test]
    fn replicate_pad_rejects_empty_padded_dim() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(&[] as &[f32], &[0], &device).unwrap();
        let err = client
            .pad_mode(&x, &[1, 0], PadMode::Replicate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dim 0"), "{err}");
        assert!(client.pad_mode(&x, &[0, 0], PadMode::Replicate).is_ok());
    }
}
