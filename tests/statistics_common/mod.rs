//! Common test helpers for statistics tests

#![allow(dead_code)]

pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

pub fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}
