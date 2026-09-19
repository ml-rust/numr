//! Per-device runtime tuning for CUDA kernel schedule choices.
//!
//! A schedule choice picks between launch alternatives that produce
//! identical bits for the same inputs. The pick is speed only. `tuned`
//! runs a probe once per device and key, caches the winner, and falls
//! back to the constant measured on one part when tuning is off or the
//! probe fails.

pub mod cache;
pub mod once;
pub mod timing;

pub use cache::{get, insert};
pub use once::{enabled, tuned};
pub use timing::time_launches;
