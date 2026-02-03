//! WGSL shader generation for multi-dtype support
//!
//! WebGPU's WGSL does not support templates like CUDA/C++.
//! This module generates WGSL shader source code for each dtype.
//!
//! # Supported DTypes
//!
//! | DType | WGSL Type | Notes |
//! |-------|-----------|-------|
//! | F32   | f32       | Always available |
//! | I32   | i32       | Always available |
//! | U32   | u32       | Always available |
//! | F16   | f16       | Requires WebGPU f16 extension |
//!
//! # Architecture
//!
//! ```text
//! generate_binary_shader(DType::F32, "add") → WGSL source with f32 types
//! generate_binary_shader(DType::I32, "add") → WGSL source with i32 types
//! generate_binary_shader(DType::U32, "add") → WGSL source with u32 types
//! ```
//!
//! Shaders are cached by `(dtype, operation)` key in the pipeline cache.

pub mod activation;
pub mod binary;
pub mod cast;
pub mod cat;
pub mod common;
pub mod compare;
pub mod complex;
pub mod cumulative;
pub mod distributions;
pub mod fft;
pub mod index;
pub mod masked;
pub mod matmul;
pub mod matrix_funcs;
pub mod norm;
pub mod reduce;
pub mod scalar;
pub mod sort;
#[cfg(feature = "sparse")]
pub mod sparse_linalg;
pub mod special;
pub mod unary;
pub mod utility;
pub mod where_cond;

pub use activation::generate_clamp_shader;
pub use binary::generate_binary_shader;
pub use cast::{generate_all_casts_from, generate_cast_shader};
pub use cat::{
    generate_cat_shader, generate_pad_shader, generate_repeat_shader, generate_roll_shader,
};
pub use common::{dtype_suffix, is_wgpu_supported, is_wgsl_float, is_wgsl_int, wgsl_type};
pub use compare::generate_compare_shader;
pub use complex::{
    complex_output_dtype, generate_angle_shader, generate_conj_shader, generate_imag_shader,
    generate_real_shader, get_complex_shader_generator, validate_complex_dtype,
};
pub use cumulative::{
    generate_cumprod_shader, generate_cumprod_strided_shader, generate_cumsum_shader,
    generate_cumsum_strided_shader, generate_logsumexp_shader, generate_logsumexp_strided_shader,
};
pub use distributions::{
    generate_bernoulli_shader, generate_beta_dist_shader, generate_binomial_shader,
    generate_chi_squared_shader, generate_exponential_shader, generate_f_distribution_shader,
    generate_gamma_dist_shader, generate_laplace_shader, generate_multinomial_count_shader,
    generate_poisson_shader, generate_student_t_shader,
};
pub use fft::{
    MAX_WORKGROUP_FFT_SIZE, generate_copy_complex_shader, generate_fftshift_shader,
    generate_hermitian_extend_shader, generate_irfft_unpack_shader, generate_rfft_pack_shader,
    generate_rfft_truncate_shader, generate_stockham_fft_shader,
};
pub use index::{
    generate_embedding_lookup_shader, generate_gather_shader, generate_index_put_shader,
    generate_index_select_shader, generate_scatter_shader, generate_validate_indices_shader,
};
pub use masked::{generate_masked_fill_shader, generate_masked_select_shader};
pub use matmul::{generate_matmul_bias_shader, generate_matmul_shader};
pub use matrix_funcs::{
    generate_diagonal_func_shader, generate_parlett_column_shader,
    generate_validate_eigenvalues_shader,
};
pub use norm::generate_norm_shader;
pub use reduce::generate_reduce_shader;
pub use scalar::{generate_fill_shader, generate_scalar_shader};
pub use sort::{
    MAX_SHARED_SORT_SIZE, generate_count_nonzero_shader, generate_flat_to_multi_index_shader,
    generate_gather_nonzero_shader, generate_searchsorted_shader, generate_sort_shader,
    generate_topk_shader, generate_unique_shader,
};
#[cfg(feature = "sparse")]
pub use sparse_linalg::{
    generate_copy_shader, generate_find_diag_indices_shader, generate_ic0_level_shader,
    generate_ilu0_level_shader, generate_sparse_trsv_lower_shader,
    generate_sparse_trsv_upper_shader,
};
pub use special::{
    generate_special_binary_shader, generate_special_ternary_shader, generate_special_unary_shader,
};
pub use unary::generate_unary_shader;
pub use utility::{
    generate_arange_shader, generate_eye_shader, generate_linspace_shader,
    generate_multinomial_with_replacement_shader, generate_multinomial_without_replacement_shader,
    generate_rand_shader, generate_randint_shader, generate_randn_shader,
};
pub use where_cond::generate_where_cond_shader;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgsl_type() {
        assert_eq!(wgsl_type(crate::dtype::DType::F32).unwrap(), "f32");
        assert_eq!(wgsl_type(crate::dtype::DType::I32).unwrap(), "i32");
        assert_eq!(wgsl_type(crate::dtype::DType::U32).unwrap(), "u32");
        assert!(wgsl_type(crate::dtype::DType::F64).is_err()); // Not supported
    }

    #[test]
    fn test_generate_binary_shader() {
        let shader = generate_binary_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn add_f32"));
        assert!(shader.contains("fn sub_f32"));
        assert!(shader.contains("fn mul_f32"));
        assert!(shader.contains("array<f32>"));
    }

    #[test]
    fn test_generate_binary_shader_i32() {
        let shader = generate_binary_shader(crate::dtype::DType::I32).unwrap();
        assert!(shader.contains("fn add_i32"));
        assert!(shader.contains("array<i32>"));
    }

    #[test]
    fn test_generate_unary_shader_float() {
        let shader = generate_unary_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn sqrt_f32"));
        assert!(shader.contains("fn exp_f32"));
        assert!(shader.contains("fn relu_f32"));
    }

    #[test]
    fn test_generate_unary_shader_int() {
        let shader = generate_unary_shader(crate::dtype::DType::I32).unwrap();
        assert!(shader.contains("fn neg_i32"));
        assert!(shader.contains("fn abs_i32"));
        // Float ops should not be present
        assert!(!shader.contains("fn sqrt_i32"));
        assert!(!shader.contains("fn exp_i32"));
    }

    #[test]
    fn test_generate_reduce_shader() {
        let shader = generate_reduce_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn reduce_sum_f32"));
        assert!(shader.contains("fn reduce_max_f32"));
        assert!(shader.contains("fn reduce_min_f32"));
    }

    #[test]
    fn test_generate_matmul_shader() {
        let shader = generate_matmul_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn matmul_f32"));
        assert!(shader.contains("fn batched_matmul_f32"));
        assert!(shader.contains("tile_a"));
        assert!(shader.contains("tile_b"));
    }

    #[test]
    fn test_generate_matmul_bias_shader() {
        let shader = generate_matmul_bias_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn matmul_bias_f32"));
        assert!(shader.contains("fn batched_matmul_bias_f32"));
        assert!(shader.contains("matmul_bias")); // bias buffer binding
        assert!(shader.contains("tile_a"));
        assert!(shader.contains("tile_b"));
        // Verify fused epilogue pattern
        assert!(shader.contains("sum + matmul_bias[col]"));
    }

    #[test]
    fn test_generate_norm_shader() {
        let shader = generate_norm_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn rms_norm_f32"));
        assert!(shader.contains("fn layer_norm_f32"));
    }

    #[test]
    fn test_generate_norm_shader_int_fails() {
        // Normalization is only for float types
        assert!(generate_norm_shader(crate::dtype::DType::I32).is_err());
    }

    #[test]
    fn test_generate_compare_shader() {
        let shader = generate_compare_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn eq_f32"));
        assert!(shader.contains("fn lt_f32"));
        assert!(shader.contains("array<f32>")); // Output is f32
    }

    // ========================================================================
    // Multi-DType WGSL Syntax Validation Tests
    //
    // These tests validate that generated shaders are syntactically correct
    // WGSL by parsing them with naga. This catches issues like:
    // - Float literals in integer contexts (0.0 vs 0)
    // - Invalid type casts
    // - Missing/incorrect array types
    // ========================================================================

    /// Helper to validate WGSL shader syntax using naga parser (re-exported by wgpu)
    fn validate_wgsl_syntax(source: &str) -> std::result::Result<(), String> {
        use wgpu::naga::front::wgsl;
        let mut frontend = wgsl::Frontend::new();
        frontend
            .parse(source)
            .map(|_| ())
            .map_err(|e| format!("WGSL parse error: {e}"))
    }

    /// All dtypes that WebGPU supports
    const WGPU_DTYPES: &[crate::dtype::DType] = &[
        crate::dtype::DType::F32,
        crate::dtype::DType::I32,
        crate::dtype::DType::U32,
    ];

    #[test]
    fn test_binary_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_binary_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate binary shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for binary shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_unary_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_unary_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate unary shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for unary shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_scalar_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_scalar_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate scalar shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for scalar shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_reduce_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_reduce_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate reduce shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for reduce shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_compare_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_compare_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate compare shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for compare shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_matmul_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_matmul_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate matmul shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for matmul shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_matmul_bias_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_matmul_bias_shader(dtype).unwrap_or_else(|_| {
                panic!("Failed to generate matmul_bias shader for {:?}", dtype)
            });
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for matmul_bias shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_norm_shader_syntax_float_only() {
        // Norm operations only support float types
        let shader = generate_norm_shader(crate::dtype::DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for norm shader F32:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_fill_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_fill_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate fill shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for fill shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_integer_shaders_no_float_literals() {
        // Verify integer shaders don't contain float literals that would cause type errors
        for dtype in [crate::dtype::DType::I32, crate::dtype::DType::U32] {
            let unary = generate_unary_shader(dtype).unwrap();
            // Integer shaders should not contain standalone float operations
            // The float ops (sqrt, exp, etc.) should be excluded for integers
            assert!(
                !unary.contains("fn sqrt_"),
                "Integer unary shader should not contain sqrt for {:?}",
                dtype
            );
            assert!(
                !unary.contains("fn exp_"),
                "Integer unary shader should not contain exp for {:?}",
                dtype
            );
        }
    }

    #[test]
    fn test_generate_cast_shader() {
        // F32 -> I32
        let shader =
            generate_cast_shader(crate::dtype::DType::F32, crate::dtype::DType::I32).unwrap();
        assert!(shader.contains("fn cast_f32_to_i32"));
        assert!(shader.contains("array<f32>"));
        assert!(shader.contains("array<i32>"));

        // I32 -> F32
        let shader =
            generate_cast_shader(crate::dtype::DType::I32, crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn cast_i32_to_f32"));

        // U32 -> F32
        let shader =
            generate_cast_shader(crate::dtype::DType::U32, crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn cast_u32_to_f32"));
    }

    #[test]
    fn test_cast_shader_syntax_all_combinations() {
        let dtypes = [
            crate::dtype::DType::F32,
            crate::dtype::DType::I32,
            crate::dtype::DType::U32,
        ];

        for &src in &dtypes {
            for &dst in &dtypes {
                if src == dst {
                    continue;
                }

                let shader = generate_cast_shader(src, dst).unwrap_or_else(|_| {
                    panic!("Failed to generate cast shader for {:?} -> {:?}", src, dst)
                });

                validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                    panic!(
                        "Invalid WGSL for cast {:?} -> {:?}:\n{}\n\nShader:\n{}",
                        src, dst, e, shader
                    )
                });
            }
        }
    }

    #[test]
    fn test_cast_shader_same_type_is_noop() {
        let shader =
            generate_cast_shader(crate::dtype::DType::F32, crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("No-op"));
        assert!(!shader.contains("@compute"));
    }

    // ========================================================================
    // Utility Operation Shader Tests (arange, linspace, eye)
    // ========================================================================

    #[test]
    fn test_generate_arange_shader_f32() {
        let shader = generate_arange_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn arange_f32"));
        assert!(shader.contains("array<f32>"));
        assert!(shader.contains("arange_params"));
    }

    #[test]
    fn test_arange_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_arange_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate arange shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for arange shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_generate_linspace_shader_f32() {
        let shader = generate_linspace_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn linspace_f32"));
        assert!(shader.contains("array<f32>"));
        assert!(shader.contains("linspace_params"));
    }

    #[test]
    fn test_linspace_shader_syntax() {
        // linspace only supports float types
        let shader = generate_linspace_shader(crate::dtype::DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for linspace shader F32:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_linspace_shader_int_fails() {
        // linspace should fail for integer types
        assert!(generate_linspace_shader(crate::dtype::DType::I32).is_err());
        assert!(generate_linspace_shader(crate::dtype::DType::U32).is_err());
    }

    #[test]
    fn test_generate_eye_shader_f32() {
        let shader = generate_eye_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn eye_f32"));
        assert!(shader.contains("array<f32>"));
        assert!(shader.contains("eye_params"));
    }

    #[test]
    fn test_eye_shader_syntax_all_dtypes() {
        for &dtype in WGPU_DTYPES {
            let shader = generate_eye_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate eye shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for eye shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    // ========================================================================
    // Random Operation Shader Tests (rand, randn, randint)
    // ========================================================================

    #[test]
    fn test_generate_rand_shader_f32() {
        let shader = generate_rand_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn rand_f32"));
        assert!(shader.contains("array<f32>"));
        assert!(shader.contains("rand_params"));
        assert!(shader.contains("pcg_hash"));
    }

    #[test]
    fn test_rand_shader_syntax() {
        // rand only supports F32 on WebGPU
        let shader = generate_rand_shader(crate::dtype::DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for rand shader F32:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_rand_shader_int_fails() {
        // rand should fail for integer types
        assert!(generate_rand_shader(crate::dtype::DType::I32).is_err());
        assert!(generate_rand_shader(crate::dtype::DType::U32).is_err());
    }

    #[test]
    fn test_generate_randn_shader_f32() {
        let shader = generate_randn_shader(crate::dtype::DType::F32).unwrap();
        assert!(shader.contains("fn randn_f32"));
        assert!(shader.contains("array<f32>"));
        assert!(shader.contains("randn_params"));
        assert!(shader.contains("box_muller"));
    }

    #[test]
    fn test_randn_shader_syntax() {
        // randn only supports F32 on WebGPU
        let shader = generate_randn_shader(crate::dtype::DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for randn shader F32:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_randn_shader_int_fails() {
        // randn should fail for integer types
        assert!(generate_randn_shader(crate::dtype::DType::I32).is_err());
        assert!(generate_randn_shader(crate::dtype::DType::U32).is_err());
    }

    #[test]
    fn test_generate_randint_shader_i32() {
        let shader = generate_randint_shader(crate::dtype::DType::I32).unwrap();
        assert!(shader.contains("fn randint_i32"));
        assert!(shader.contains("array<i32>"));
        assert!(shader.contains("randint_params"));
    }

    #[test]
    fn test_generate_randint_shader_u32() {
        let shader = generate_randint_shader(crate::dtype::DType::U32).unwrap();
        assert!(shader.contains("fn randint_u32"));
        assert!(shader.contains("array<u32>"));
    }

    #[test]
    fn test_randint_shader_syntax_int_dtypes() {
        // randint supports I32 and U32
        for dtype in [crate::dtype::DType::I32, crate::dtype::DType::U32] {
            let shader = generate_randint_shader(dtype)
                .unwrap_or_else(|_| panic!("Failed to generate randint shader for {:?}", dtype));
            validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
                panic!(
                    "Invalid WGSL for randint shader {:?}:\n{}\n\nShader:\n{}",
                    dtype, e, shader
                )
            });
        }
    }

    #[test]
    fn test_randint_shader_float_fails() {
        // randint should fail for float types
        assert!(generate_randint_shader(crate::dtype::DType::F32).is_err());
    }

    // ========================================================================
    // Special Function Shader Tests
    // ========================================================================

    #[test]
    fn test_special_unary_shader_syntax() {
        let shader = generate_special_unary_shader(crate::dtype::DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for special unary shader F32:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_special_binary_shader_syntax() {
        let shader = generate_special_binary_shader(crate::dtype::DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for special binary shader F32:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_special_ternary_shader_syntax() {
        let shader = generate_special_ternary_shader(crate::dtype::DType::F32).unwrap();
        validate_wgsl_syntax(&shader).unwrap_or_else(|e| {
            panic!(
                "Invalid WGSL for special ternary shader F32:\n{}\n\nShader:\n{}",
                e, shader
            )
        });
    }

    #[test]
    fn test_special_shaders_f64_fails() {
        // Special functions only support F32 on WebGPU (no F64)
        assert!(generate_special_unary_shader(crate::dtype::DType::F64).is_err());
        assert!(generate_special_binary_shader(crate::dtype::DType::F64).is_err());
        assert!(generate_special_ternary_shader(crate::dtype::DType::F64).is_err());
    }
}
