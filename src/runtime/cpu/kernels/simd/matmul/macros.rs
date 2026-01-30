//! Macros for generating SIMD microkernels
//!
//! These macros eliminate code duplication between AVX2 and AVX-512 implementations.
//! Each macro generates a microkernel with the same algorithm but different SIMD intrinsics.

/// Generate a 6×NR matmul microkernel for f32
///
/// Parameters:
/// - `$name`: Function name (e.g., `microkernel_6x16_f32`)
/// - `$nr`: Column width (8 for AVX2, 16 for AVX-512)
/// - `$feat1`, `$feat2`: Target features (e.g., "avx512f", "fma")
/// - `$loadu`: Unaligned load intrinsic
/// - `$storeu`: Unaligned store intrinsic
/// - `$set1`: Broadcast intrinsic
/// - `$fmadd`: Fused multiply-add intrinsic
/// - `$reg_ty`: Register type (e.g., `__m256` or `__m512`)
macro_rules! define_microkernel_f32 {
    (
        $name:ident,
        $nr:expr,
        $feat1:literal,
        $feat2:literal,
        $loadu:ident,
        $storeu:ident,
        $set1:ident,
        $fmadd:ident,
        $reg_ty:ty
    ) => {
        /// Matmul microkernel: C[0:6, 0:NR] += A[0:6, 0:K] @ B[0:K, 0:NR]
        ///
        /// # Safety
        /// - All pointers must be valid for the specified dimensions
        /// - CPU must support the required SIMD features
        #[target_feature(enable = $feat1)]
        #[target_feature(enable = $feat2)]
        pub unsafe fn $name(a: *const f32, b: *const f32, c: *mut f32, k: usize, ldc: usize) {
            // Load C accumulators (6 rows)
            let mut c0 = $loadu(c);
            let mut c1 = $loadu(c.add(ldc));
            let mut c2 = $loadu(c.add(ldc * 2));
            let mut c3 = $loadu(c.add(ldc * 3));
            let mut c4 = $loadu(c.add(ldc * 4));
            let mut c5 = $loadu(c.add(ldc * 5));

            for kk in 0..k {
                let b_row = $loadu(b.add(kk * $nr));
                let a_base = a.add(kk * 6);

                let a0 = $set1(*a_base);
                c0 = $fmadd(a0, b_row, c0);

                let a1 = $set1(*a_base.add(1));
                c1 = $fmadd(a1, b_row, c1);

                let a2 = $set1(*a_base.add(2));
                c2 = $fmadd(a2, b_row, c2);

                let a3 = $set1(*a_base.add(3));
                c3 = $fmadd(a3, b_row, c3);

                let a4 = $set1(*a_base.add(4));
                c4 = $fmadd(a4, b_row, c4);

                let a5 = $set1(*a_base.add(5));
                c5 = $fmadd(a5, b_row, c5);
            }

            $storeu(c, c0);
            $storeu(c.add(ldc), c1);
            $storeu(c.add(ldc * 2), c2);
            $storeu(c.add(ldc * 3), c3);
            $storeu(c.add(ldc * 4), c4);
            $storeu(c.add(ldc * 5), c5);
        }
    };
}

/// Generate a 6×NR matmul microkernel for f64
macro_rules! define_microkernel_f64 {
    (
        $name:ident,
        $nr:expr,
        $feat1:literal,
        $feat2:literal,
        $loadu:ident,
        $storeu:ident,
        $set1:ident,
        $fmadd:ident,
        $reg_ty:ty
    ) => {
        /// Matmul microkernel: C[0:6, 0:NR] += A[0:6, 0:K] @ B[0:K, 0:NR]
        ///
        /// # Safety
        /// - All pointers must be valid for the specified dimensions
        /// - CPU must support the required SIMD features
        #[target_feature(enable = $feat1)]
        #[target_feature(enable = $feat2)]
        pub unsafe fn $name(a: *const f64, b: *const f64, c: *mut f64, k: usize, ldc: usize) {
            let mut c0 = $loadu(c);
            let mut c1 = $loadu(c.add(ldc));
            let mut c2 = $loadu(c.add(ldc * 2));
            let mut c3 = $loadu(c.add(ldc * 3));
            let mut c4 = $loadu(c.add(ldc * 4));
            let mut c5 = $loadu(c.add(ldc * 5));

            for kk in 0..k {
                let b_row = $loadu(b.add(kk * $nr));
                let a_base = a.add(kk * 6);

                let a0 = $set1(*a_base);
                c0 = $fmadd(a0, b_row, c0);

                let a1 = $set1(*a_base.add(1));
                c1 = $fmadd(a1, b_row, c1);

                let a2 = $set1(*a_base.add(2));
                c2 = $fmadd(a2, b_row, c2);

                let a3 = $set1(*a_base.add(3));
                c3 = $fmadd(a3, b_row, c3);

                let a4 = $set1(*a_base.add(4));
                c4 = $fmadd(a4, b_row, c4);

                let a5 = $set1(*a_base.add(5));
                c5 = $fmadd(a5, b_row, c5);
            }

            $storeu(c, c0);
            $storeu(c.add(ldc), c1);
            $storeu(c.add(ldc * 2), c2);
            $storeu(c.add(ldc * 3), c3);
            $storeu(c.add(ldc * 4), c4);
            $storeu(c.add(ldc * 5), c5);
        }
    };
}

pub(crate) use define_microkernel_f32;
pub(crate) use define_microkernel_f64;
