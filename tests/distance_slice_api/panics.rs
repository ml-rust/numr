//! Length-mismatch panics of the free functions and the held handle.

use numr::distance::{self, Kernels};

#[test]
#[should_panic(expected = "dot_f32: length mismatch: a.len() = 3, b.len() = 4")]
fn dot_f32_length_mismatch_panics() {
    distance::dot_f32(&[1.0; 3], &[1.0; 4]);
}

#[test]
#[should_panic(expected = "cosine_distance_f64: length mismatch: a.len() = 2, b.len() = 0")]
fn cosine_f64_length_mismatch_panics() {
    distance::cosine_distance_f64(&[1.0; 2], &[]);
}

#[test]
#[should_panic(expected = "l2_squared_f32: length mismatch: a.len() = 5, b.len() = 1")]
fn held_kernels_length_mismatch_panics() {
    Kernels::scalar().l2_squared_f32(&[1.0; 5], &[1.0; 1]);
}

#[test]
#[should_panic(expected = "dot_i8: length mismatch: a.len() = 1, b.len() = 2")]
fn dot_i8_length_mismatch_panics() {
    distance::dot_i8(&[1], &[1, 2]);
}

#[test]
#[should_panic(expected = "dot_many_f32: query.len() = 2 does not equal d = 3")]
fn many_query_length_mismatch_panics() {
    let mut out = [0.0f32; 2];
    distance::dot_many_f32(&[1.0; 2], &[1.0; 6], 3, &mut out);
}

#[test]
#[should_panic(
    expected = "manhattan_many_f64: rows.len() = 5 does not equal out.len() * d = 2 * 3"
)]
fn many_rows_length_mismatch_panics() {
    let mut out = [0.0f64; 2];
    distance::manhattan_many_f64(&[1.0; 3], &[1.0; 5], 3, &mut out);
}

#[test]
#[should_panic(
    expected = "cosine_distance_many_f32: rows.len() = 3 does not equal out.len() * d = 2 * 0"
)]
fn many_zero_d_with_rows_panics() {
    let mut out = [0.0f32; 2];
    distance::cosine_distance_many_f32(&[], &[1.0; 3], 0, &mut out);
}

#[test]
#[should_panic(expected = "dot_i8_many: rows.len() = 7 does not equal out.len() * d = 3 * 2")]
fn i8_many_rows_length_mismatch_panics() {
    let mut out = [0i32; 3];
    distance::dot_i8_many(&[1, 2], &[1; 7], 2, &mut out);
}
