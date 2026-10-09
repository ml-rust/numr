//! Result type of the one-pass cosine reduction.

/// The three sums cosine distance needs, gathered in one pass.
///
/// The caller turns them into a distance. `metrics::cosine` owns that formula,
/// including its `denom == 0 -> 0` rule.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CosineSums<T> {
    /// `sum(a[i] * b[i])`.
    pub dot: T,
    /// `sum(a[i] * a[i])`.
    pub norm_a: T,
    /// `sum(b[i] * b[i])`.
    pub norm_b: T,
}
