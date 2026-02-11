//! Macros for generating SIMD microkernels
//!
//! These macros eliminate code duplication between AVX2 and AVX-512 implementations.
//! Each macro generates a microkernel with the same algorithm but different SIMD intrinsics.
//!
//! # Beta parameter (first_k)
//!
//! When `first_k = true` (first K-block), accumulators start from zero (setzero)
//! instead of loading from C. This eliminates the separate zero-pass over the output
//! matrix, saving a full write+read cache pollution pass.
//!
//! # Double-width microkernels (6×2NR)
//!
//! Process 2 column chunks per row to get 12 independent FMA chains (6 rows × 2 chunks).
//! FMA latency=4, throughput=0.5 → need 8+ chains to saturate. 12 > 8, so pipeline is full.
//! Each k iteration: 2 B loads shared across 6 A broadcasts = good reuse.

/// Generate a 6×NR matmul microkernel for f32 (single column chunk)
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
        $setzero:ident,
        $reg_ty:ty
    ) => {
        /// Matmul microkernel: C[0:6, 0:NR] += A[0:6, 0:K] @ B[0:K, 0:NR]
        #[target_feature(enable = $feat1)]
        #[target_feature(enable = $feat2)]
        pub unsafe fn $name(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            k: usize,
            ldc: usize,
            first_k: bool,
        ) {
            let mut c0: $reg_ty;
            let mut c1: $reg_ty;
            let mut c2: $reg_ty;
            let mut c3: $reg_ty;
            let mut c4: $reg_ty;
            let mut c5: $reg_ty;

            if first_k {
                c0 = $setzero();
                c1 = $setzero();
                c2 = $setzero();
                c3 = $setzero();
                c4 = $setzero();
                c5 = $setzero();
            } else {
                c0 = $loadu(c);
                c1 = $loadu(c.add(ldc));
                c2 = $loadu(c.add(ldc * 2));
                c3 = $loadu(c.add(ldc * 3));
                c4 = $loadu(c.add(ldc * 4));
                c5 = $loadu(c.add(ldc * 5));
            }

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

/// Generate a 6×(2*NR) double-width matmul microkernel for f32
///
/// Processes 2 column chunks per row = 12 independent FMA chains.
macro_rules! define_microkernel_2x_f32 {
    (
        $name:ident,
        $nr:expr,
        $feat1:literal,
        $feat2:literal,
        $loadu:ident,
        $storeu:ident,
        $set1:ident,
        $fmadd:ident,
        $setzero:ident,
        $reg_ty:ty
    ) => {
        /// Matmul microkernel: C[0:6, 0:2*NR] += A[0:6, 0:K] @ B[0:K, 0:2*NR]
        ///
        /// Double-width: 6 rows × 2 column chunks = 12 accumulators.
        #[target_feature(enable = $feat1)]
        #[target_feature(enable = $feat2)]
        pub unsafe fn $name(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            k: usize,
            ldc: usize,
            first_k: bool,
        ) {
            // 12 accumulators: 6 rows × 2 column chunks
            let (mut c00, mut c01): ($reg_ty, $reg_ty);
            let (mut c10, mut c11): ($reg_ty, $reg_ty);
            let (mut c20, mut c21): ($reg_ty, $reg_ty);
            let (mut c30, mut c31): ($reg_ty, $reg_ty);
            let (mut c40, mut c41): ($reg_ty, $reg_ty);
            let (mut c50, mut c51): ($reg_ty, $reg_ty);

            let nr2 = 2 * $nr;

            if first_k {
                c00 = $setzero();
                c01 = $setzero();
                c10 = $setzero();
                c11 = $setzero();
                c20 = $setzero();
                c21 = $setzero();
                c30 = $setzero();
                c31 = $setzero();
                c40 = $setzero();
                c41 = $setzero();
                c50 = $setzero();
                c51 = $setzero();
            } else {
                c00 = $loadu(c);
                c01 = $loadu(c.add($nr));
                c10 = $loadu(c.add(ldc));
                c11 = $loadu(c.add(ldc + $nr));
                c20 = $loadu(c.add(ldc * 2));
                c21 = $loadu(c.add(ldc * 2 + $nr));
                c30 = $loadu(c.add(ldc * 3));
                c31 = $loadu(c.add(ldc * 3 + $nr));
                c40 = $loadu(c.add(ldc * 4));
                c41 = $loadu(c.add(ldc * 4 + $nr));
                c50 = $loadu(c.add(ldc * 5));
                c51 = $loadu(c.add(ldc * 5 + $nr));
            }

            for kk in 0..k {
                // Load 2 B vectors (shared across 6 rows)
                let b0 = $loadu(b.add(kk * nr2));
                let b1 = $loadu(b.add(kk * nr2 + $nr));
                let a_base = a.add(kk * 6);

                let a0 = $set1(*a_base);
                c00 = $fmadd(a0, b0, c00);
                c01 = $fmadd(a0, b1, c01);

                let a1 = $set1(*a_base.add(1));
                c10 = $fmadd(a1, b0, c10);
                c11 = $fmadd(a1, b1, c11);

                let a2 = $set1(*a_base.add(2));
                c20 = $fmadd(a2, b0, c20);
                c21 = $fmadd(a2, b1, c21);

                let a3 = $set1(*a_base.add(3));
                c30 = $fmadd(a3, b0, c30);
                c31 = $fmadd(a3, b1, c31);

                let a4 = $set1(*a_base.add(4));
                c40 = $fmadd(a4, b0, c40);
                c41 = $fmadd(a4, b1, c41);

                let a5 = $set1(*a_base.add(5));
                c50 = $fmadd(a5, b0, c50);
                c51 = $fmadd(a5, b1, c51);
            }

            $storeu(c, c00);
            $storeu(c.add($nr), c01);
            $storeu(c.add(ldc), c10);
            $storeu(c.add(ldc + $nr), c11);
            $storeu(c.add(ldc * 2), c20);
            $storeu(c.add(ldc * 2 + $nr), c21);
            $storeu(c.add(ldc * 3), c30);
            $storeu(c.add(ldc * 3 + $nr), c31);
            $storeu(c.add(ldc * 4), c40);
            $storeu(c.add(ldc * 4 + $nr), c41);
            $storeu(c.add(ldc * 5), c50);
            $storeu(c.add(ldc * 5 + $nr), c51);
        }
    };
}

/// Generate a 6×NR matmul microkernel for f64 (single column chunk)
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
        $setzero:ident,
        $reg_ty:ty
    ) => {
        /// Matmul microkernel: C[0:6, 0:NR] += A[0:6, 0:K] @ B[0:K, 0:NR]
        #[target_feature(enable = $feat1)]
        #[target_feature(enable = $feat2)]
        pub unsafe fn $name(
            a: *const f64,
            b: *const f64,
            c: *mut f64,
            k: usize,
            ldc: usize,
            first_k: bool,
        ) {
            let mut c0: $reg_ty;
            let mut c1: $reg_ty;
            let mut c2: $reg_ty;
            let mut c3: $reg_ty;
            let mut c4: $reg_ty;
            let mut c5: $reg_ty;

            if first_k {
                c0 = $setzero();
                c1 = $setzero();
                c2 = $setzero();
                c3 = $setzero();
                c4 = $setzero();
                c5 = $setzero();
            } else {
                c0 = $loadu(c);
                c1 = $loadu(c.add(ldc));
                c2 = $loadu(c.add(ldc * 2));
                c3 = $loadu(c.add(ldc * 3));
                c4 = $loadu(c.add(ldc * 4));
                c5 = $loadu(c.add(ldc * 5));
            }

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

/// Generate a 6×(2*NR) double-width matmul microkernel for f64
macro_rules! define_microkernel_2x_f64 {
    (
        $name:ident,
        $nr:expr,
        $feat1:literal,
        $feat2:literal,
        $loadu:ident,
        $storeu:ident,
        $set1:ident,
        $fmadd:ident,
        $setzero:ident,
        $reg_ty:ty
    ) => {
        /// Matmul microkernel: C[0:6, 0:2*NR] += A[0:6, 0:K] @ B[0:K, 0:2*NR]
        #[target_feature(enable = $feat1)]
        #[target_feature(enable = $feat2)]
        pub unsafe fn $name(
            a: *const f64,
            b: *const f64,
            c: *mut f64,
            k: usize,
            ldc: usize,
            first_k: bool,
        ) {
            let (mut c00, mut c01): ($reg_ty, $reg_ty);
            let (mut c10, mut c11): ($reg_ty, $reg_ty);
            let (mut c20, mut c21): ($reg_ty, $reg_ty);
            let (mut c30, mut c31): ($reg_ty, $reg_ty);
            let (mut c40, mut c41): ($reg_ty, $reg_ty);
            let (mut c50, mut c51): ($reg_ty, $reg_ty);

            let nr2 = 2 * $nr;

            if first_k {
                c00 = $setzero();
                c01 = $setzero();
                c10 = $setzero();
                c11 = $setzero();
                c20 = $setzero();
                c21 = $setzero();
                c30 = $setzero();
                c31 = $setzero();
                c40 = $setzero();
                c41 = $setzero();
                c50 = $setzero();
                c51 = $setzero();
            } else {
                c00 = $loadu(c);
                c01 = $loadu(c.add($nr));
                c10 = $loadu(c.add(ldc));
                c11 = $loadu(c.add(ldc + $nr));
                c20 = $loadu(c.add(ldc * 2));
                c21 = $loadu(c.add(ldc * 2 + $nr));
                c30 = $loadu(c.add(ldc * 3));
                c31 = $loadu(c.add(ldc * 3 + $nr));
                c40 = $loadu(c.add(ldc * 4));
                c41 = $loadu(c.add(ldc * 4 + $nr));
                c50 = $loadu(c.add(ldc * 5));
                c51 = $loadu(c.add(ldc * 5 + $nr));
            }

            for kk in 0..k {
                let b0 = $loadu(b.add(kk * nr2));
                let b1 = $loadu(b.add(kk * nr2 + $nr));
                let a_base = a.add(kk * 6);

                let a0 = $set1(*a_base);
                c00 = $fmadd(a0, b0, c00);
                c01 = $fmadd(a0, b1, c01);

                let a1 = $set1(*a_base.add(1));
                c10 = $fmadd(a1, b0, c10);
                c11 = $fmadd(a1, b1, c11);

                let a2 = $set1(*a_base.add(2));
                c20 = $fmadd(a2, b0, c20);
                c21 = $fmadd(a2, b1, c21);

                let a3 = $set1(*a_base.add(3));
                c30 = $fmadd(a3, b0, c30);
                c31 = $fmadd(a3, b1, c31);

                let a4 = $set1(*a_base.add(4));
                c40 = $fmadd(a4, b0, c40);
                c41 = $fmadd(a4, b1, c41);

                let a5 = $set1(*a_base.add(5));
                c50 = $fmadd(a5, b0, c50);
                c51 = $fmadd(a5, b1, c51);
            }

            $storeu(c, c00);
            $storeu(c.add($nr), c01);
            $storeu(c.add(ldc), c10);
            $storeu(c.add(ldc + $nr), c11);
            $storeu(c.add(ldc * 2), c20);
            $storeu(c.add(ldc * 2 + $nr), c21);
            $storeu(c.add(ldc * 3), c30);
            $storeu(c.add(ldc * 3 + $nr), c31);
            $storeu(c.add(ldc * 4), c40);
            $storeu(c.add(ldc * 4 + $nr), c41);
            $storeu(c.add(ldc * 5), c50);
            $storeu(c.add(ldc * 5 + $nr), c51);
        }
    };
}

pub(crate) use define_microkernel_2x_f32;
pub(crate) use define_microkernel_2x_f64;
pub(crate) use define_microkernel_f32;
pub(crate) use define_microkernel_f64;
