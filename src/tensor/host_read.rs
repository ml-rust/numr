//! Width check shared by every device-to-host read.

use crate::dtype::DataType;
use crate::error::{Error, Result};

/// An error unless host type `T` has the width of one `dtype` element.
///
/// The copy size is `numel * size_of::<T>()`. A wider `T` reads past the
/// buffer, and a narrower `T` returns a truncated view. A block-quantized
/// dtype has no per-element width, so every typed read of it is an error.
pub(crate) fn check_host_read<D: DataType, T>(dtype: D) -> Result<()> {
    let elem_bytes = std::mem::size_of::<T>();
    let dtype_bytes = dtype.size_in_bytes();
    if dtype.is_quantized() || dtype_bytes != elem_bytes {
        return Err(Error::HostReadTypeMismatch {
            dtype: dtype.short_name(),
            dtype_bytes,
            elem_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::error::Error;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::{Storage, Tensor};

    fn is_mismatch<T>(result: crate::error::Result<T>, bytes: usize, elem: usize) -> bool {
        matches!(
            result,
            Err(Error::HostReadTypeMismatch { dtype_bytes, elem_bytes, .. })
                if dtype_bytes == bytes && elem_bytes == elem
        )
    }

    /// A 2-byte tensor read as `f32` would copy twice its buffer.
    ///
    /// Sabotage check: remove the check from `Tensor::try_to_vec` and the
    /// read returns `Ok` with bytes from past the allocation.
    #[test]
    fn tensor_read_of_a_narrower_dtype_is_an_error() {
        let device = CpuDevice::new();
        let bits = [0x3f80u16, 0x4000, 0x4040];
        let bytes: &[u8] = bytemuck::cast_slice(&bits);
        let tensor = Tensor::<CpuRuntime>::from_bytes(bytes, &[3], DType::BF16, &device)
            .expect("bf16 tensor builds");

        assert!(is_mismatch(tensor.try_to_vec::<f32>(), 2, 4));
        assert_eq!(tensor.try_to_vec::<u16>().ok(), Some(bits.to_vec()));
    }

    /// A narrowed view keeps its element offset in the dtype's own width.
    #[test]
    fn narrowed_read_in_the_native_width_returns_the_last_row() {
        let device = CpuDevice::new();
        let bits: Vec<u16> = (0..6).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&bits);
        let tensor = Tensor::<CpuRuntime>::from_bytes(bytes, &[1, 2, 3], DType::F16, &device)
            .expect("f16 tensor builds");
        let row = tensor.narrow(-2, 1, 1).expect("narrow runs");

        assert!(is_mismatch(row.try_to_vec::<f32>(), 2, 4));
        assert_eq!(row.try_to_vec::<u16>().ok(), Some(vec![3, 4, 5]));
    }

    /// A wider dtype read as a narrower type is an error, not a short read.
    #[test]
    fn tensor_read_of_a_wider_dtype_is_an_error() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device)
            .expect("f32 tensor builds");

        assert!(is_mismatch(tensor.try_to_vec::<u8>(), 4, 1));
        assert!(is_mismatch(tensor.try_to_vec::<f64>(), 4, 8));
        assert_eq!(tensor.try_to_vec::<f32>().ok(), Some(vec![1.0, 2.0]));
    }

    /// The pipelined read applies the same width check.
    #[test]
    fn pipelined_read_of_a_mismatched_dtype_is_an_error() {
        let device = CpuDevice::new();
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device)
            .expect("f32 tensor builds");
        let event = tensor.record_event().expect("event records");

        assert!(is_mismatch(tensor.to_vec_pipelined::<f64>(event), 4, 8));
    }

    /// `item` reads one element at the dtype's offset, so it checks the width.
    #[test]
    fn item_of_a_mismatched_dtype_is_an_error() {
        let device = CpuDevice::new();
        let bytes: &[u8] = bytemuck::cast_slice(&[0x3f80u16]);
        let scalar = Tensor::<CpuRuntime>::from_bytes(bytes, &[], DType::BF16, &device)
            .expect("bf16 scalar builds");

        assert!(is_mismatch(scalar.item::<f32>(), 2, 4));
        assert_eq!(scalar.item::<u16>().ok(), Some(0x3f80));
    }

    /// `Storage::try_to_vec` sizes its copy from the element count too.
    #[test]
    fn storage_read_of_a_mismatched_dtype_is_an_error() {
        let device = CpuDevice::new();
        let bits = [1u16, 2, 3, 4];
        let bytes: &[u8] = bytemuck::cast_slice(&bits);
        let storage =
            Storage::<CpuRuntime>::from_bytes(bytes, DType::F16, &device).expect("storage builds");

        assert!(is_mismatch(storage.try_to_vec::<f32>(), 2, 4));
        assert_eq!(storage.try_to_vec::<u16>().ok(), Some(bits.to_vec()));
    }
}
