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

pub mod binary;
pub mod cast;
pub mod cat;
pub mod common;
pub mod compare;
pub mod index;
pub mod masked;
pub mod matmul;
pub mod norm;
pub mod reduce;
pub mod scalar;
pub mod unary;
pub mod utility;

pub use binary::generate_binary_shader;
pub use cast::{generate_all_casts_from, generate_cast_shader};
pub use cat::generate_cat_shader;
pub use common::{dtype_suffix, is_wgpu_supported, is_wgsl_float, is_wgsl_int, wgsl_type};
pub use compare::generate_compare_shader;
pub use index::{generate_gather_shader, generate_index_select_shader, generate_scatter_shader};
pub use masked::{generate_masked_fill_shader, generate_masked_select_shader};
pub use matmul::generate_matmul_shader;
pub use norm::generate_norm_shader;
pub use reduce::generate_reduce_shader;
pub use scalar::{generate_fill_shader, generate_scalar_shader};
pub use unary::generate_unary_shader;
pub use utility::{generate_arange_shader, generate_eye_shader, generate_linspace_shader};

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
}
