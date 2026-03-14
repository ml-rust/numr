//! CPU kernels for 2:4 structured sparsity
//!
//! Low-level kernels for pruning to 2:4 format, decompression, and sparse matmul.

use crate::dtype::Element;

/// Prune a dense [M, K] matrix to 2:4 structured sparsity.
///
/// For each group of 4 elements along K, keeps the 2 with largest magnitude.
///
/// # Arguments
/// * `dense` - Input data, row-major [M, K]
/// * `compressed` - Output compressed values, row-major [M, K/2]
/// * `metadata` - Output packed metadata, row-major [M, meta_cols] as u32
/// * `m` - Number of rows
/// * `k` - Number of columns (must be divisible by 4)
///
/// # Safety
/// Caller must ensure all pointers are valid and buffers are correctly sized.
pub unsafe fn prune_to_24_kernel<T: Element>(
    dense: *const T,
    compressed: *mut T,
    metadata: *mut u32,
    m: usize,
    k: usize,
) {
    let num_groups = k / 4;
    let meta_cols = (num_groups + 7) / 8;
    let half_k = k / 2;

    for row in 0..m {
        let row_in = dense.add(row * k);
        let row_out = compressed.add(row * half_k);
        let row_meta = metadata.add(row * meta_cols);

        // Zero out metadata
        for mc in 0..meta_cols {
            *row_meta.add(mc) = 0;
        }

        let mut out_idx = 0usize;

        for g in 0..num_groups {
            let base = g * 4;
            let vals = [
                *row_in.add(base),
                *row_in.add(base + 1),
                *row_in.add(base + 2),
                *row_in.add(base + 3),
            ];

            // Compute magnitudes and find top-2
            let mags: [f64; 4] = [
                vals[0].to_f64().abs(),
                vals[1].to_f64().abs(),
                vals[2].to_f64().abs(),
                vals[3].to_f64().abs(),
            ];

            // Find the 2 largest magnitudes (stable: prefer earlier indices on tie)
            let (idx0, idx1) = top_2_indices(&mags);

            // Write compressed values (lower index first)
            let (first, second) = if idx0 < idx1 {
                (idx0, idx1)
            } else {
                (idx1, idx0)
            };
            *row_out.add(out_idx) = vals[first];
            *row_out.add(out_idx + 1) = vals[second];
            out_idx += 2;

            // Build 4-bit bitmask: bit i set means position i is kept
            let mask: u32 = (1 << first) | (1 << second);

            // Pack into metadata word
            let word_idx = g / 8;
            let nibble_idx = g % 8;
            let word = row_meta.add(word_idx);
            *word |= mask << (nibble_idx * 4);
        }
    }
}

/// Find indices of the 2 largest values in a 4-element array.
/// On ties, prefers earlier indices.
#[inline]
fn top_2_indices(mags: &[f64; 4]) -> (usize, usize) {
    // Simple approach: find max, then find second max
    let mut indices = [0usize, 1, 2, 3];
    // Sort by magnitude descending, stable (preserves order on ties)
    indices.sort_by(|&a, &b| {
        mags[b]
            .partial_cmp(&mags[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (indices[0], indices[1])
}

/// Decompress a 2:4 sparse tensor back to dense format.
///
/// # Arguments
/// * `compressed` - Input compressed values, row-major [M, K/2]
/// * `metadata` - Input packed metadata, row-major [M, meta_cols] as u32
/// * `dense` - Output dense values, row-major [M, K]
/// * `m` - Number of rows
/// * `k` - Number of columns
///
/// # Safety
/// Caller must ensure all pointers are valid and buffers are correctly sized.
pub unsafe fn decompress_24_kernel<T: Element>(
    compressed: *const T,
    metadata: *const u32,
    dense: *mut T,
    m: usize,
    k: usize,
) {
    let num_groups = k / 4;
    let meta_cols = (num_groups + 7) / 8;
    let half_k = k / 2;
    let zero = T::zeroed();

    for row in 0..m {
        let row_in = compressed.add(row * half_k);
        let row_meta = metadata.add(row * meta_cols);
        let row_out = dense.add(row * k);

        let mut in_idx = 0usize;

        for g in 0..num_groups {
            let base = g * 4;
            let word_idx = g / 8;
            let nibble_idx = g % 8;
            let word = *row_meta.add(word_idx);
            let mask = (word >> (nibble_idx * 4)) & 0xF;

            // Write zeros first, then overwrite kept positions
            for i in 0..4 {
                *row_out.add(base + i) = zero;
            }

            // Place the 2 compressed values at their original positions
            for bit in 0..4u32 {
                if mask & (1 << bit) != 0 {
                    *row_out.add(base + bit as usize) = *row_in.add(in_idx);
                    in_idx += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_roundtrip_f32() {
        // Dense matrix: 2x8
        let dense: Vec<f32> = vec![
            1.0, -3.0, 2.0, 0.5, // group 0: keep -3.0 (idx1), 2.0 (idx2)
            0.1, 0.2, 0.3, 0.4, // group 1: keep 0.3 (idx2), 0.4 (idx3)
            4.0, 1.0, -5.0, 3.0, // group 2: keep 4.0 (idx0), -5.0 (idx2)
            0.0, 0.0, 0.0, 0.0, // group 3: all zero, keep first 2
        ];
        let m = 2;
        let k = 8;
        let half_k = k / 2;
        let meta_cols = 1; // 2 groups per row, fits in 1 u32

        let mut compressed = vec![0.0f32; m * half_k];
        let mut metadata = vec![0u32; m * meta_cols];

        unsafe {
            prune_to_24_kernel(
                dense.as_ptr(),
                compressed.as_mut_ptr(),
                metadata.as_mut_ptr(),
                m,
                k,
            );
        }

        // Verify: group 0 (row 0): -3.0 (idx1) and 2.0 (idx2) are top-2
        // compressed[0] should be -3.0 (idx1), compressed[1] should be 2.0 (idx2)
        // (sorted by index: idx1 < idx2)
        assert_eq!(compressed[0], -3.0);
        assert_eq!(compressed[1], 2.0);

        // Now decompress and verify roundtrip
        let mut reconstructed = vec![0.0f32; m * k];
        unsafe {
            decompress_24_kernel(
                compressed.as_ptr(),
                metadata.as_ptr(),
                reconstructed.as_mut_ptr(),
                m,
                k,
            );
        }

        // Row 0, group 0: positions 1,2 kept → [0, -3, 2, 0]
        assert_eq!(reconstructed[0], 0.0);
        assert_eq!(reconstructed[1], -3.0);
        assert_eq!(reconstructed[2], 2.0);
        assert_eq!(reconstructed[3], 0.0);
    }

    #[test]
    fn test_top_2_indices() {
        // Basic case
        assert_eq!(top_2_indices(&[1.0, 3.0, 2.0, 0.5]), (1, 2));
        // Ties: prefer earlier indices
        assert_eq!(top_2_indices(&[1.0, 1.0, 1.0, 1.0]), (0, 1));
        // Negative magnitudes (should not happen since we pass abs, but test anyway)
        assert_eq!(top_2_indices(&[0.0, 0.0, 0.0, 0.0]), (0, 1));
    }
}
