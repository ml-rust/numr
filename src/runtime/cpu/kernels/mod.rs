//! CPU kernel implementations
//!
//! This module provides low-level compute kernels for CPU operations.
//! Kernels are generic over `T: Element` and dispatch based on operation type.

#![allow(unsafe_op_in_unsafe_fn)] // Kernels are already marked unsafe, inner unsafe is redundant

pub mod advanced_random;
pub mod binary;
pub mod binary_int;
pub mod compare;
pub mod complex;
pub mod conv;
pub mod conv_transpose;
pub mod cumulative;
pub mod distance;
pub mod distributions;
pub mod fft;
pub mod fused_add_norm;
pub mod fused_elementwise;
pub mod gemm_epilogue;
pub mod index;
pub mod ipow;
pub mod logical;
pub mod matmul;
pub mod matmul_i8;
pub mod memory;
pub mod norm;
pub mod quasirandom;
pub mod reduce;
pub(crate) mod rng;
pub mod scalar;
pub mod scatter_reduce_int;
pub mod semiring_matmul;
pub mod simd;
pub mod sobol_data;
pub mod sort;
#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "sparse")]
pub mod sparse_24;
pub mod unary;
pub mod where_select;
pub mod wide_acc;

// Re-export all kernel functions for convenient access
pub use advanced_random::{
    pcg64_randn_kernel, pcg64_uniform_kernel, philox_randn_kernel, philox_uniform_kernel,
    threefry_randn_kernel, threefry_uniform_kernel, xoshiro256_randn_kernel,
    xoshiro256_uniform_kernel,
};
pub use binary::{binary_op_kernel, binary_op_strided_kernel};
pub use compare::{compare_op_kernel, compare_op_strided_kernel};
pub use complex::{
    angle_complex64, angle_complex128, angle_real_f32, angle_real_f64, complex64_div_real,
    complex64_mul_real, complex128_div_real, complex128_mul_real, conj_complex64, conj_complex128,
    from_real_imag_f32, from_real_imag_f64, imag_complex64, imag_complex128, real_complex64,
    real_complex128,
};
pub use conv::{conv1d_kernel, conv2d_kernel, depthwise_conv2d_kernel};
pub use conv_transpose::conv_transpose1d_kernel;
pub use cumulative::{
    cumprod_kernel, cumprod_strided_kernel, cumsum_kernel, cumsum_strided_kernel, logsumexp_kernel,
    logsumexp_strided_kernel,
};
pub use distance::{cdist_kernel, pdist_kernel, squareform_inverse_kernel, squareform_kernel};
pub use distributions::{
    bernoulli_kernel, beta_kernel, binomial_kernel, chi_squared_kernel, exponential_kernel,
    f_distribution_kernel, gamma_kernel, laplace_kernel, poisson_kernel, student_t_kernel,
};
pub use fft::{
    fftshift_c64, fftshift_c128, ifftshift_c64, ifftshift_c128, irfft_c64, irfft_c128, rfft_c64,
    rfft_c128, stockham_fft_batched_c64, stockham_fft_batched_c128,
};
pub use fused_add_norm::{
    fused_add_layer_norm_bwd_kernel, fused_add_layer_norm_kernel, fused_add_rms_norm_bwd_kernel,
    fused_add_rms_norm_kernel,
};
pub use fused_elementwise::{
    fused_add_mul_kernel, fused_mul_add_kernel, fused_mul_add_scalar_kernel,
};
pub use gemm_epilogue::{
    matmul_bias_activation_bwd_kernel, matmul_bias_activation_kernel, matmul_bias_residual_kernel,
};
pub use index::{
    bincount_kernel, embedding_lookup_kernel, gather_2d_kernel, gather_kernel, gather_nd_kernel,
    index_put_kernel, index_select_kernel, masked_fill_kernel, masked_select_kernel,
    max_i64_kernel, scatter_kernel, scatter_reduce_kernel, slice_assign_kernel,
};
pub use logical::{logical_and_kernel, logical_not_kernel, logical_or_kernel, logical_xor_kernel};
pub use matmul::{
    gemv_bt_kernel, gemv_bt_wide_kernel, matmul_bias_kernel, matmul_bt_kernel,
    matmul_bt_matches_contiguous, matmul_kernel, matmul_wide_kernel,
};
pub use matmul_i8::{matmul_i8_to_i32_bias_kernel, matmul_i8_to_i32_kernel};
pub use memory::{
    arange_kernel, cast_kernel, copy_kernel, eye_kernel, fill_kernel, linspace_kernel,
    multinomial_kernel_with_replacement, multinomial_kernel_without_replacement, one_hot_kernel,
    rand_normal_kernel, rand_uniform_kernel, randint_kernel, randperm_kernel,
};
pub use norm::{group_norm_kernel, layer_norm_kernel, rms_norm_kernel};
pub use quasirandom::{
    halton_f32, halton_f64, latin_hypercube_f32, latin_hypercube_f64, sobol_f32, sobol_f64,
};
pub use reduce::{
    Accumulator, argmax_kernel, argmin_kernel, reduce_kernel, reduce_kernel_with_precision,
    softmax_bwd_kernel, softmax_kernel, variance_kernel,
};
pub use scalar::{rsub_scalar_kernel, scalar_op_kernel};
pub use sort::{
    argsort_kernel, count_nonzero_kernel, count_unique_kernel, extract_unique_kernel,
    flat_to_multi_index_kernel, nonzero_flat_kernel, searchsorted_kernel, sort_kernel,
    sort_values_kernel, topk_kernel, unique_with_counts_kernel,
};
pub use unary::{
    clamp_kernel, elu_kernel, gelu_kernel, gelu_mul_kernel, isinf_kernel, isnan_kernel,
    leaky_relu_kernel, relu_kernel, relu_mul_kernel, sigmoid_kernel, sigmoid_mul_kernel,
    silu_kernel, silu_mul_kernel, snake_beta_bwd_kernel, snake_beta_kernel, unary_op_kernel,
};
pub use where_select::{
    where_kernel, where_kernel_generic, where_strided_kernel, where_strided_kernel_generic,
};

// Re-export SIMD dot product kernels for downstream crates (e.g., boostr quantized ops)
#[allow(unused_imports)]
pub use simd::dot::{i8xi8_dot_f32, i8xi8_dot_i32};

// Re-export sparse kernel functions for external use
#[cfg(feature = "sparse")]
#[allow(unused_imports)]
pub use sparse::{
    divide_by_pivot, find_pivot, find_pivot_range, gather_and_clear, gather_and_clear_i32,
    scatter_column, scatter_column_i32, sparse_axpy, sparse_axpy_i32, swap_rows,
};
