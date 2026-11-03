//! Helpers shared by the strided copy shaders.
//!
//! `copy` and `copy_into` pack the same kernel parameters and share the 2-D
//! thread index. Each helper takes the operation name, so an error still names
//! the operation that raised it.

use super::copy::MAX_DIMS;
use crate::error::{Error, Result};

/// WGSL constants and `thread_index` for a kernel on the 2-D dispatch grid.
///
/// Concatenate it into a shader source once. Both copy shaders do.
pub(super) const THREAD_INDEX_WGSL: &str = r#"
const GRID_X: u32 = 65535u;
const WORKGROUP_SIZE: u32 = 256u;

fn thread_index(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * GRID_X * WORKGROUP_SIZE;
}
"#;

/// A `BackendLimitation` on WebGPU for `operation`.
pub(super) fn limitation(operation: &'static str, reason: String) -> Error {
    Error::BackendLimitation {
        backend: "wgpu",
        operation,
        reason,
    }
}

/// Narrow `value` to u32, or name `what` in a `BackendLimitation`.
pub(super) fn to_u32(operation: &'static str, value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        limitation(
            operation,
            format!("{what} {value} exceeds the 32-bit addressing of the copy kernel"),
        )
    })
}

/// Pack a shape into kernel words.
pub(super) fn shape_words(
    operation: &'static str,
    shape: &[usize],
    out: &mut [u32; MAX_DIMS],
) -> Result<()> {
    for (slot, &dim) in out.iter_mut().zip(shape) {
        *slot = to_u32(operation, dim, "dimension")?;
    }
    Ok(())
}

/// Pack a stride list into kernel words. Negative strides wrap as u32.
pub(super) fn stride_words(
    operation: &'static str,
    strides: &[isize],
    out: &mut [u32; MAX_DIMS],
) -> Result<()> {
    for (slot, &stride) in out.iter_mut().zip(strides) {
        let stride = i32::try_from(stride)
            .map_err(|_| limitation(operation, format!("stride {stride} does not fit 32 bits")))?;
        // The kernel adds strides as wrapping u32, which equals i32 addition.
        *slot = stride as u32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_name_the_calling_operation() {
        let Err(Error::BackendLimitation { operation, .. }) = to_u32("copy_into", usize::MAX, "x")
        else {
            panic!("expected a BackendLimitation");
        };
        assert_eq!(operation, "copy_into");
    }

    #[test]
    fn negative_stride_wraps_to_two_complement() {
        let mut words = [0u32; MAX_DIMS];
        assert!(stride_words("copy_strided", &[-1, 3], &mut words).is_ok());
        assert_eq!(words[0], u32::MAX);
        assert_eq!(words[1], 3);
    }

    #[test]
    fn shape_words_reject_a_dimension_over_32_bits() {
        let mut words = [0u32; MAX_DIMS];
        if usize::BITS > 32 {
            assert!(shape_words("copy_strided", &[usize::MAX], &mut words).is_err());
        }
        assert!(shape_words("copy_strided", &[2, 5], &mut words).is_ok());
        assert_eq!(&words[..2], &[2, 5]);
    }
}
