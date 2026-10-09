//! Length and shape checks shared by every public distance function.

/// Panics unless the two input lengths match.
#[track_caller]
#[inline]
pub(super) fn assert_same_len(op: &str, a_len: usize, b_len: usize) {
    assert!(
        a_len == b_len,
        "{op}: length mismatch: a.len() = {a_len}, b.len() = {b_len}"
    );
}

/// Checks the one-query-many-rows shape, then writes `f(row)` into each slot
/// of `out`.
///
/// `rows` holds `out.len()` rows of `d` components, row-major. With `d == 0`,
/// `rows` must be empty and every row is the empty slice.
///
/// # Panics
/// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
#[track_caller]
#[inline]
pub(super) fn for_each_row<T, R>(
    op: &str,
    query: &[T],
    rows: &[T],
    d: usize,
    out: &mut [R],
    mut f: impl FnMut(&[T]) -> R,
) {
    assert!(
        query.len() == d,
        "{op}: query.len() = {} does not equal d = {d}",
        query.len()
    );
    assert!(
        out.len().checked_mul(d) == Some(rows.len()),
        "{op}: rows.len() = {} does not equal out.len() * d = {} * {d}",
        rows.len(),
        out.len()
    );
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = f(&rows[i * d..(i + 1) * d]);
    }
}
