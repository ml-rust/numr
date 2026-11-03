//! Strided-to-strided copy on CPU, the kernel behind `copy_into` with a
//! non-contiguous destination.
//!
//! The loop walks the logical indices of the source shape in row-major order.
//! Each element moves from its source position to the matching destination
//! position.
//!
//! [`coalesce_n`] first merges adjacent dimensions that walk memory as one in
//! both the source and the destination. When the innermost dimension then has
//! stride 1 in both, each inner row is one contiguous run and moves with a
//! single `copy_nonoverlapping`. Otherwise the inner dimension moves one
//! element at a time. The outer dimensions advance an odometer that tracks
//! both element offsets.

use super::strided_copy::coalesce_n;

/// Copy `shape`-many elements from a strided source to a strided destination.
///
/// Strides are in elements and can be negative. `src` and `dst` point at the
/// first element of each view.
///
/// # Safety
///
/// - Every element offset reachable from `shape` and the matching strides
///   must lie inside the allocation behind `src` (reads) and `dst` (writes).
/// - `src_strides`, `dst_strides` and `shape` have the same length.
/// - The element sets the two views touch must not overlap.
pub(crate) unsafe fn copy_strided_into_impl(
    src: *const u8,
    src_strides: &[isize],
    dst: *mut u8,
    dst_strides: &[isize],
    shape: &[usize],
    elem_size: usize,
) {
    if shape.contains(&0) || elem_size == 0 {
        return;
    }
    // A rank-0 view coalesces to one single-element dimension.
    let (shape, [src_strides, dst_strides]) = coalesce_n(shape, [src_strides, dst_strides]);
    let outer = shape.len() - 1;
    let inner = shape[outer];
    let (inner_src, inner_dst) = (src_strides[outer], dst_strides[outer]);
    let es = elem_size as isize;

    // A unit stride on both sides makes the inner row one contiguous run.
    // Otherwise each element moves on its own.
    let walk = if inner_src == 1 && inner_dst == 1 {
        Walk {
            chunks: 1,
            chunk_bytes: inner * elem_size,
            step_src: 0,
            step_dst: 0,
        }
    } else {
        Walk {
            chunks: inner as isize,
            chunk_bytes: elem_size,
            step_src: inner_src * es,
            step_dst: inner_dst * es,
        }
    };
    let outer_shape = &shape[..outer];
    let (src_outer, dst_outer) = (&src_strides[..outer], &dst_strides[..outer]);

    // A fixed chunk size lets the compiler emit one load and one store per
    // element. Contiguous rows and other element sizes use a sized memcpy.
    // SAFETY: forwarded from this function's contract.
    unsafe {
        match (walk.chunks, walk.chunk_bytes) {
            (1, _) => walk_rows::<0>(src, dst, es, &walk, outer_shape, src_outer, dst_outer),
            (_, 1) => walk_rows::<1>(src, dst, es, &walk, outer_shape, src_outer, dst_outer),
            (_, 2) => walk_rows::<2>(src, dst, es, &walk, outer_shape, src_outer, dst_outer),
            (_, 4) => walk_rows::<4>(src, dst, es, &walk, outer_shape, src_outer, dst_outer),
            (_, 8) => walk_rows::<8>(src, dst, es, &walk, outer_shape, src_outer, dst_outer),
            (_, 16) => walk_rows::<16>(src, dst, es, &walk, outer_shape, src_outer, dst_outer),
            _ => walk_rows::<0>(src, dst, es, &walk, outer_shape, src_outer, dst_outer),
        }
    }
}

/// How each inner row moves: `chunks` copies of `chunk_bytes`, stepping
/// `step_src` and `step_dst` bytes between copies.
struct Walk {
    chunks: isize,
    chunk_bytes: usize,
    step_src: isize,
    step_dst: isize,
}

/// Move every inner row, advancing the odometer over the outer dimensions.
///
/// `N` is the chunk size in bytes when it is known at compile time. `N == 0`
/// copies `walk.chunk_bytes` with a sized memcpy.
///
/// # Safety
///
/// Same contract as [`copy_strided_into_impl`]. When `N != 0`,
/// `walk.chunk_bytes == N`.
#[allow(clippy::too_many_arguments)]
unsafe fn walk_rows<const N: usize>(
    src: *const u8,
    dst: *mut u8,
    es: isize,
    walk: &Walk,
    outer_shape: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) {
    let rows: usize = outer_shape.iter().product();
    let mut idx = vec![0usize; outer_shape.len()];
    let (mut src_off, mut dst_off) = (0isize, 0isize);
    for _ in 0..rows {
        for j in 0..walk.chunks {
            // SAFETY: the caller guarantees every reachable offset is in
            // bounds and the element sets are disjoint. This chunk starts at
            // element offset `src_off` (`dst_off`) plus `j` inner steps and
            // spans `chunk_bytes`, which stays inside the inner row.
            unsafe {
                let s = src.offset(src_off * es + j * walk.step_src);
                let d = dst.offset(dst_off * es + j * walk.step_dst);
                if N == 0 {
                    std::ptr::copy_nonoverlapping(s, d, walk.chunk_bytes);
                } else {
                    let v = std::ptr::read_unaligned(s as *const [u8; N]);
                    std::ptr::write_unaligned(d as *mut [u8; N], v);
                }
            }
        }
        advance(
            &mut idx,
            &mut src_off,
            &mut dst_off,
            outer_shape,
            src_strides,
            dst_strides,
        );
    }
}

/// Advance a row-major odometer over the outer dimensions.
///
/// `idx` holds the current outer index. `src_off` and `dst_off` hold the
/// matching element offsets and move with it.
fn advance(
    idx: &mut [usize],
    src_off: &mut isize,
    dst_off: &mut isize,
    shape: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) {
    for d in (0..idx.len()).rev() {
        idx[d] += 1;
        *src_off += src_strides[d];
        *dst_off += dst_strides[d];
        if idx[d] < shape[d] {
            return;
        }
        let span = shape[d] as isize;
        *src_off -= src_strides[d] * span;
        *dst_off -= dst_strides[d] * span;
        idx[d] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Copy `src` (laid out by `src_strides`) into a zeroed buffer through
    /// `dst_strides`, starting `dst_start` elements into the buffer.
    fn run(
        src: &[u32],
        src_strides: &[isize],
        dst_len: usize,
        dst_start: usize,
        dst_strides: &[isize],
        shape: &[usize],
    ) -> Vec<u32> {
        let mut dst = vec![0u32; dst_len];
        // SAFETY: the callers below pass strides whose reach stays inside both buffers.
        unsafe {
            copy_strided_into_impl(
                src.as_ptr() as *const u8,
                src_strides,
                dst.as_mut_ptr().add(dst_start) as *mut u8,
                dst_strides,
                shape,
                4,
            );
        }
        dst
    }

    #[test]
    fn writes_a_channel_slice() {
        // Source [2, 2] into columns 1..3 of a [2, 4] buffer.
        let dst = run(&[1, 2, 3, 4], &[2, 1], 8, 1, &[4, 1], &[2, 2]);
        assert_eq!(dst, vec![0, 1, 2, 0, 0, 3, 4, 0]);
    }

    #[test]
    fn writes_through_a_permuted_destination() {
        // Destination strides [1, 2] write the transpose of a [2, 3] source.
        let dst = run(&[1, 2, 3, 4, 5, 6], &[3, 1], 6, 0, &[1, 2], &[2, 3]);
        assert_eq!(dst, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn reads_a_negative_stride_source() {
        // Source walks the buffer backwards from its last element.
        let mut dst = vec![0u32; 3];
        let src = [1u32, 2, 3];
        // SAFETY: the source starts at index 2 and steps back by 1 for 3 elements.
        unsafe {
            copy_strided_into_impl(
                src.as_ptr().add(2) as *const u8,
                &[-1],
                dst.as_mut_ptr() as *mut u8,
                &[1],
                &[3],
                4,
            );
        }
        assert_eq!(dst, vec![3, 2, 1]);
    }

    #[test]
    fn copies_coalesced_rows_into_a_strided_destination() {
        // A contiguous [2, 3, 4] source into a destination whose rows of 12
        // are contiguous but 16 apart. The inner two dimensions coalesce into
        // one run of 12, so each row moves with a single copy.
        let src: Vec<u32> = (1..=24).collect();
        let dst = run(&src, &[12, 4, 1], 28, 0, &[16, 4, 1], &[2, 3, 4]);
        let mut expected = vec![0u32; 28];
        expected[..12].copy_from_slice(&src[..12]);
        expected[16..28].copy_from_slice(&src[12..]);
        assert_eq!(dst, expected);
    }

    #[test]
    fn copies_a_destination_that_cannot_coalesce() {
        // Row-major source into a transposed destination: neither dimension
        // merges and the inner destination stride is 3, so elements move singly.
        let src: Vec<u32> = (1..=12).collect();
        let dst = run(&src, &[4, 1], 12, 0, &[1, 3], &[3, 4]);
        assert_eq!(dst, vec![1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]);
    }

    #[test]
    fn copies_a_rank_zero_view() {
        let dst = run(&[9], &[], 1, 0, &[], &[]);
        assert_eq!(dst, vec![9]);
    }

    #[test]
    fn skips_an_empty_view() {
        let dst = run(&[], &[1, 1], 4, 0, &[2, 1], &[2, 0]);
        assert_eq!(dst, vec![0; 4]);
    }

    #[test]
    fn copies_odd_element_sizes() {
        let src = [1u8, 2, 3, 4, 5, 6];
        let mut dst = [0u8; 6];
        // SAFETY: two 3-byte elements fill both 6-byte buffers exactly.
        unsafe {
            copy_strided_into_impl(src.as_ptr(), &[1], dst.as_mut_ptr(), &[1], &[2], 3);
        }
        assert_eq!(dst, [1, 2, 3, 4, 5, 6]);
    }
}
