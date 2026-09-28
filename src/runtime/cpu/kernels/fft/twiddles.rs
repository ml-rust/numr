//! Per-stage twiddle tables for the radix-2 Stockham kernels.
//!
//! Every entry uses the exact f64 angle expression the butterfly loop used
//! before the table existed, `sign * 2.0 * PI * (b as f64) / (m as f64)`, and
//! the same `cos`/`sin` calls. A table-driven transform is therefore
//! bit-identical to one that evaluates the twiddle inside the butterfly.

use crate::dtype::{Complex64, Complex128};
use std::f64::consts::PI;

/// Twiddles `W_m^b = exp(sign * 2*pi*i * b / m)` for every stage of an N-point transform.
///
/// Stage `s` has `half_m = 2^s` butterflies per group and owns the entries
/// `[half_m - 1, 2 * half_m - 1)`. The table holds `N - 1` entries in total.
pub(super) struct StockhamTwiddles<T> {
    n: usize,
    table: Vec<T>,
}

impl<T: Copy> StockhamTwiddles<T> {
    fn build(n: usize, inverse: bool, narrow: impl Fn(f64) -> T) -> Self {
        debug_assert!(n > 0 && n.is_power_of_two(), "N must be power of 2");
        let sign = if inverse { 1.0f64 } else { -1.0f64 };
        let log_n = n.trailing_zeros() as usize;
        let mut table = Vec::with_capacity(n.saturating_sub(1));
        for stage in 0..log_n {
            let m: usize = 1 << (stage + 1);
            let half_m: usize = 1 << stage;
            for b in 0..half_m {
                let theta = sign * 2.0 * PI * (b as f64) / (m as f64);
                table.push(narrow(theta));
            }
        }
        Self { n, table }
    }

    /// Transform length the table was built for.
    pub(super) fn n(&self) -> usize {
        self.n
    }

    /// Twiddles of the stage with `half_m` butterflies per group, indexed by `b`.
    #[inline(always)]
    pub(super) fn stage(&self, half_m: usize) -> &[T] {
        &self.table[half_m - 1..2 * half_m - 1]
    }
}

impl StockhamTwiddles<Complex64> {
    /// Table for the f32 kernel. Each value narrows the f64 `cos`/`sin` result.
    pub(super) fn new_c64(n: usize, inverse: bool) -> Self {
        Self::build(n, inverse, |theta| {
            Complex64::new(theta.cos() as f32, theta.sin() as f32)
        })
    }
}

impl StockhamTwiddles<Complex128> {
    /// Table for the f64 kernel.
    pub(super) fn new_c128(n: usize, inverse: bool) -> Self {
        Self::build(n, inverse, |theta| {
            Complex128::new(theta.cos(), theta.sin())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_matches_inline_expression_bitwise() {
        for &n in &[1usize, 2, 4, 64, 1024] {
            for &inverse in &[false, true] {
                let t64 = StockhamTwiddles::new_c64(n, inverse);
                let t128 = StockhamTwiddles::new_c128(n, inverse);
                assert_eq!(t64.table.len(), n - 1);
                let sign = if inverse { 1.0f64 } else { -1.0f64 };
                let log_n = n.trailing_zeros() as usize;
                for stage in 0..log_n {
                    let m: usize = 1 << (stage + 1);
                    let half_m: usize = 1 << stage;
                    for b in 0..half_m {
                        let theta = sign * 2.0 * PI * (b as f64) / (m as f64);
                        let w64 = t64.stage(half_m)[b];
                        let w128 = t128.stage(half_m)[b];
                        assert_eq!(w64.re.to_bits(), (theta.cos() as f32).to_bits());
                        assert_eq!(w64.im.to_bits(), (theta.sin() as f32).to_bits());
                        assert_eq!(w128.re.to_bits(), theta.cos().to_bits());
                        assert_eq!(w128.im.to_bits(), theta.sin().to_bits());
                    }
                }
            }
        }
    }
}
