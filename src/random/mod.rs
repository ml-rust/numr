//! Seed control for random ops that take no explicit seed.

pub mod generator;

pub use generator::{manual_seed, next_call_seed, seed_state, set_seed_state, with_seed};
