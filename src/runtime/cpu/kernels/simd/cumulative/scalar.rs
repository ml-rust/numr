//! Scalar fallbacks for strided cumulative operations.

#[inline]
pub(super) unsafe fn cumsum_strided_scalar_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = 0.0f32;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc += *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

#[inline]
pub(super) unsafe fn cumsum_strided_scalar_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = 0.0f64;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc += *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

#[inline]
pub(super) unsafe fn cumprod_strided_scalar_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = 1.0f32;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc *= *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

#[inline]
pub(super) unsafe fn cumprod_strided_scalar_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = 1.0f64;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc *= *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}
