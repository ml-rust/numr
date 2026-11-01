//! COO sort kernel launchers for the CUDA sparse format conversions.
//!
//! COO -> CSR/CSC sorts entries by a composite key on the GPU, then gathers
//! indices and values through the sorted entry ids.

mod kernels;

pub use kernels::{
    launch_coo_compute_keys, launch_coo_gather, launch_coo_init_indices,
    launch_thrust_sort_pairs_i64_i32,
};
