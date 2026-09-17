//! Scalar fallbacks for masked fill/select/count.

#[inline]
pub(super) unsafe fn masked_fill_scalar_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
    value: f32,
) {
    for i in 0..len {
        *output.add(i) = if *mask.add(i) != 0 {
            value
        } else {
            *input.add(i)
        };
    }
}

#[inline]
pub(super) unsafe fn masked_fill_scalar_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
    value: f64,
) {
    for i in 0..len {
        *output.add(i) = if *mask.add(i) != 0 {
            value
        } else {
            *input.add(i)
        };
    }
}

#[inline]
pub(super) unsafe fn masked_select_scalar_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
) -> usize {
    let mut out_idx = 0;
    for i in 0..len {
        if *mask.add(i) != 0 {
            *output.add(out_idx) = *input.add(i);
            out_idx += 1;
        }
    }
    out_idx
}

#[inline]
pub(super) unsafe fn masked_select_scalar_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
) -> usize {
    let mut out_idx = 0;
    for i in 0..len {
        if *mask.add(i) != 0 {
            *output.add(out_idx) = *input.add(i);
            out_idx += 1;
        }
    }
    out_idx
}

#[inline]
pub(super) unsafe fn masked_count_scalar(mask: *const u8, len: usize) -> usize {
    let mut count = 0;
    for i in 0..len {
        if *mask.add(i) != 0 {
            count += 1;
        }
    }
    count
}
