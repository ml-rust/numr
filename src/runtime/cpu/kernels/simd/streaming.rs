//! Non-temporal (streaming) store utilities for large arrays
//!
//! Non-temporal stores bypass the CPU cache and write directly to RAM.
//! This is beneficial for large arrays (> L3 cache) where the output
//! won't be read immediately, avoiding cache pollution.
//!
//! # When to Use
//!
//! - Arrays larger than ~1MB (larger than typical L3 cache)
//! - Write-only operations (output not read immediately after)
//! - Element-wise ops that don't reuse output data
//!
//! # Requirements
//!
//! - **Alignment**: Streaming stores require aligned memory
//!   - AVX2: 32-byte alignment for `_mm256_stream_ps`
//!   - AVX-512: 64-byte alignment for `_mm512_stream_ps`
//! - **Memory fence**: `_mm_sfence()` must be called after streaming stores
//!   to ensure writes are visible to other threads/devices
//!
//! # Performance
//!
//! Expected speedup: 10-30% on very large arrays that exceed L3 cache.
//! No benefit (possibly slight penalty) for small arrays.

/// Threshold in bytes above which streaming stores are beneficial.
/// Set to 1MB - arrays larger than this will use non-temporal stores.
pub const STREAMING_THRESHOLD_BYTES: usize = 1024 * 1024;

/// Threshold in f32 elements (1MB / 4 bytes = 262144)
pub const STREAMING_THRESHOLD_F32: usize = STREAMING_THRESHOLD_BYTES / 4;

/// Threshold in f64 elements (1MB / 8 bytes = 131072)
pub const STREAMING_THRESHOLD_F64: usize = STREAMING_THRESHOLD_BYTES / 8;

/// Check if streaming stores should be used based on array length
#[inline]
pub const fn should_stream_f32(len: usize) -> bool {
    len >= STREAMING_THRESHOLD_F32
}

/// Check if streaming stores should be used based on array length
#[inline]
pub const fn should_stream_f64(len: usize) -> bool {
    len >= STREAMING_THRESHOLD_F64
}

/// AVX2 alignment requirement (32 bytes)
pub const AVX2_ALIGN: usize = 32;

/// AVX-512 alignment requirement (64 bytes)
pub const AVX512_ALIGN: usize = 64;

/// Check if pointer is aligned for AVX2 streaming stores (32-byte)
#[inline]
pub fn is_aligned_avx2<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(AVX2_ALIGN)
}

/// Check if pointer is aligned for AVX-512 streaming stores (64-byte)
#[inline]
pub fn is_aligned_avx512<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(AVX512_ALIGN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thresholds() {
        assert_eq!(STREAMING_THRESHOLD_F32, 262144);
        assert_eq!(STREAMING_THRESHOLD_F64, 131072);
    }

    #[test]
    fn test_should_stream() {
        assert!(!should_stream_f32(100));
        assert!(!should_stream_f32(262143));
        assert!(should_stream_f32(262144));
        assert!(should_stream_f32(1_000_000));

        assert!(!should_stream_f64(100));
        assert!(!should_stream_f64(131071));
        assert!(should_stream_f64(131072));
        assert!(should_stream_f64(500_000));
    }

    #[test]
    fn test_alignment_check() {
        // Stack allocation might not be aligned, but we can test the logic
        let aligned_64: [f32; 16] = [0.0; 16];
        let ptr = aligned_64.as_ptr();

        // The alignment check should work (result depends on actual allocation)
        let _is_avx2 = is_aligned_avx2(ptr);
        let _is_avx512 = is_aligned_avx512(ptr);

        // Test with known aligned address (0 is trivially aligned to anything)
        assert!(is_aligned_avx2(std::ptr::null::<f32>()));
        assert!(is_aligned_avx512(std::ptr::null::<f32>()));
        assert!(is_aligned_avx2(32 as *const f32));
        assert!(is_aligned_avx512(64 as *const f32));
        assert!(!is_aligned_avx512(32 as *const f32)); // 32 not aligned to 64
    }
}
