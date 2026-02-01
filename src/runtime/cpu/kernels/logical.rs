//! Logical operation kernels (AND, OR, XOR, NOT)

/// Logical AND: out[i] = a[i] && b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_and_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] != 0 && b_slice[i] != 0 {
            1
        } else {
            0
        };
    }
}

/// Logical OR: out[i] = a[i] || b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_or_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] != 0 || b_slice[i] != 0 {
            1
        } else {
            0
        };
    }
}

/// Logical XOR: out[i] = a[i] ^ b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_xor_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let a_bool = a_slice[i] != 0;
        let b_bool = b_slice[i] != 0;
        out_slice[i] = if a_bool != b_bool { 1 } else { 0 };
    }
}

/// Logical NOT: out[i] = !a[i]
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_not_kernel(a: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] == 0 { 1 } else { 0 };
    }
}
