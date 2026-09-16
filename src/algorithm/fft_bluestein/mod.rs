//! Host-side Bluestein (chirp-z) tables, shared by every backend.
//!
//! Bluestein rewrites an N-point DFT as a cyclic convolution of length
//! `M = (2N - 1).next_power_of_two()`, which any radix-2 kernel evaluates
//! directly. See [`BluesteinTables`] for details.

mod bluestein;

pub use bluestein::{BluesteinTables, cached_tables, chirp_sequence};
