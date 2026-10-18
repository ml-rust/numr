//! Byte copies between caller host buffers, named by `u64` addresses, and memory the group owns.
//!
//! This is the only module of the local communicator that dereferences a caller pointer.

/// Append `len` bytes from host address `src` to `out`.
///
/// # Safety
///
/// When `len > 0`, `src` must point to `len` readable bytes of host memory.
/// No thread can write those bytes while the copy runs.
pub(super) unsafe fn read_into(src: u64, len: usize, out: &mut Vec<u8>) {
    if len > 0 {
        // SAFETY: the caller guarantees `src` names `len` readable bytes with no concurrent
        // writer. `out` is group-owned, so it cannot overlap the caller buffer.
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(src as *const u8, len) });
    }
}

/// Copy `len` bytes from host address `src` into a new vector.
///
/// # Safety
///
/// Same contract as [`read_into`].
pub(super) unsafe fn read(src: u64, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    // SAFETY: forwarded from this function's contract.
    unsafe { read_into(src, len, &mut out) };
    out
}

/// Copy `bytes` to host address `dst`.
///
/// # Safety
///
/// When `bytes` is not empty, `dst` must point to `bytes.len()` writable bytes of host memory.
/// No other thread can read or write those bytes while the copy runs.
/// `bytes` must not overlap the destination.
pub(super) unsafe fn write(dst: u64, bytes: &[u8]) {
    if !bytes.is_empty() {
        // SAFETY: the caller guarantees `dst` names `bytes.len()` writable bytes that no other
        // thread touches, and that the destination does not overlap `bytes`.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, bytes.len()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_and_write_round_trip() {
        let src = [1u8, 2, 3, 4];
        let copy = unsafe { read(src.as_ptr() as u64, src.len()) };
        assert_eq!(copy, src);

        let mut dst = [0u8; 4];
        unsafe { write(dst.as_mut_ptr() as u64, &copy) };
        assert_eq!(dst, src);
    }

    #[test]
    fn zero_length_never_touches_the_pointer() {
        assert!(unsafe { read(0, 0) }.is_empty());
        unsafe { write(0, &[]) };
    }
}
