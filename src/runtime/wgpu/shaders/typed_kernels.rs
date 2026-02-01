//! Compile-time enforced typed kernel implementations for WebGPU
//!
//! This module implements `TypedKernel<T>` for `WgpuClient` for each supported dtype.
//! If any implementation is missing, code using that dtype won't compile.
//!
//! # Supported DTypes
//!
//! - `f32` - Always available
//! - `i32` - Always available
//! - `u32` - Always available
//! - `f16` - Requires WebGPU f16 extension (future)
//!
//! # Compile-Time Enforcement
//!
//! The `WgpuKernels` trait requires all dtype implementations. If `WgpuClient`
//! implements `WgpuKernels`, then all `TypedKernel<T>` implementations must exist.
//!
//! ```ignore
//! // This line enforces that ALL required TypedKernel<T> are implemented:
//! impl WgpuKernels for WgpuClient {}
//! // If TypedKernel<i32> is missing → COMPILE ERROR
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{ComputePipeline, ShaderModule};

use super::super::client::{WgpuClient, get_buffer};
use super::generator::{
    dtype_suffix, generate_binary_shader, generate_compare_shader, generate_fill_shader,
    generate_matmul_shader, generate_norm_shader, generate_reduce_shader, generate_scalar_shader,
    generate_unary_shader,
};
use super::pipeline::{LayoutKey, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOp, ReduceOp, UnaryOp};
use crate::runtime::kernel::{
    CompareOp, TypedCompare, TypedKernel, TypedMatmul, TypedNorm, WgpuKernels,
};

// ============================================================================
// Shader Cache
// ============================================================================

/// Cache key for compiled shaders
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ShaderCacheKey {
    dtype: DType,
    category: &'static str, // "binary", "unary", "scalar", "compare", "reduce", "matmul", "norm"
}

/// Cache for compiled shader modules and pipelines per dtype
struct DTypeShaderCache {
    modules: RwLock<HashMap<ShaderCacheKey, Arc<ShaderModule>>>,
    pipelines: RwLock<HashMap<(ShaderCacheKey, &'static str), Arc<ComputePipeline>>>,
}

impl DTypeShaderCache {
    fn new() -> Self {
        Self {
            modules: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    fn get_or_create_module(
        &self,
        client: &WgpuClient,
        dtype: DType,
        category: &'static str,
        source_fn: impl FnOnce() -> Result<String>,
    ) -> Result<Arc<ShaderModule>> {
        let key = ShaderCacheKey { dtype, category };

        // Check cache first
        {
            let cache = self.modules.read();
            if let Some(module) = cache.get(&key) {
                return Ok(module.clone());
            }
        }

        // Generate and compile shader
        let source = source_fn()?;
        let module = client
            .wgpu_device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}_{}", category, dtype.short_name())),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let module = Arc::new(module);
        {
            let mut cache = self.modules.write();
            cache.insert(key, module.clone());
        }

        Ok(module)
    }

    fn get_or_create_pipeline(
        &self,
        client: &WgpuClient,
        dtype: DType,
        category: &'static str,
        entry_point: &'static str,
        module: &ShaderModule,
        layout: &wgpu::BindGroupLayout,
    ) -> Arc<ComputePipeline> {
        let key = (ShaderCacheKey { dtype, category }, entry_point);

        // Check cache first
        {
            let cache = self.pipelines.read();
            if let Some(pipeline) = cache.get(&key) {
                return pipeline.clone();
            }
        }

        // Create pipeline layout
        let pipeline_layout =
            client
                .wgpu_device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}_{}_layout", category, entry_point)),
                    bind_group_layouts: &[layout],
                    immediate_size: 0,
                });

        // Create pipeline
        let pipeline =
            client
                .wgpu_device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{}_{}", category, entry_point)),
                    layout: Some(&pipeline_layout),
                    module,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let pipeline = Arc::new(pipeline);
        {
            let mut cache = self.pipelines.write();
            cache.insert(key, pipeline.clone());
        }

        pipeline
    }
}

/// Global shader cache
static SHADER_CACHE: std::sync::OnceLock<DTypeShaderCache> = std::sync::OnceLock::new();

fn shader_cache() -> &'static DTypeShaderCache {
    SHADER_CACHE.get_or_init(DTypeShaderCache::new)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Entry point lookup table for binary operations.
/// Maps (op_name, suffix) to entry point name.
static BINARY_ENTRY_POINTS: &[(&str, &str, &str)] = &[
    ("add", "f32", "add_f32"),
    ("add", "i32", "add_i32"),
    ("add", "u32", "add_u32"),
    ("add", "f16", "add_f16"),
    ("sub", "f32", "sub_f32"),
    ("sub", "i32", "sub_i32"),
    ("sub", "u32", "sub_u32"),
    ("sub", "f16", "sub_f16"),
    ("mul", "f32", "mul_f32"),
    ("mul", "i32", "mul_i32"),
    ("mul", "u32", "mul_u32"),
    ("mul", "f16", "mul_f16"),
    ("div", "f32", "div_f32"),
    ("div", "i32", "div_i32"),
    ("div", "u32", "div_u32"),
    ("div", "f16", "div_f16"),
    ("pow", "f32", "pow_f32"),
    ("pow", "i32", "pow_i32"),
    ("pow", "u32", "pow_u32"),
    ("pow", "f16", "pow_f16"),
    ("max", "f32", "max_f32"),
    ("max", "i32", "max_i32"),
    ("max", "u32", "max_u32"),
    ("max", "f16", "max_f16"),
    ("min", "f32", "min_f32"),
    ("min", "i32", "min_i32"),
    ("min", "u32", "min_u32"),
    ("min", "f16", "min_f16"),
    ("atan2", "f32", "atan2_f32"),
    ("atan2", "f16", "atan2_f16"),
];

fn binary_op_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Pow => "pow",
        BinaryOp::Max => "max",
        BinaryOp::Min => "min",
        BinaryOp::Atan2 => "atan2",
    }
}

fn binary_op_to_entry_point(op: BinaryOp, suffix: &str) -> Result<&'static str> {
    let op_name = binary_op_name(op);

    // Look up entry point in table
    for &(name, sfx, entry) in BINARY_ENTRY_POINTS {
        if name == op_name && sfx == suffix {
            return Ok(entry);
        }
    }

    // Return error for unsupported combinations
    Err(Error::Internal(format!(
        "No binary entry point for {}_{}",
        op_name, suffix
    )))
}

/// Entry point lookup table for unary operations.
/// Maps (op_name, suffix) to entry point name.
/// Using a static lookup table is more maintainable than repeated match arms.
static UNARY_ENTRY_POINTS: &[(&str, &str, &str)] = &[
    // All-type operations: (op_name, suffix, entry_point)
    ("neg", "f32", "neg_f32"),
    ("neg", "i32", "neg_i32"),
    ("neg", "u32", "neg_u32"),
    ("neg", "f16", "neg_f16"),
    ("abs", "f32", "abs_f32"),
    ("abs", "i32", "abs_i32"),
    ("abs", "u32", "abs_u32"),
    ("abs", "f16", "abs_f16"),
    ("square", "f32", "square_f32"),
    ("square", "i32", "square_i32"),
    ("square", "u32", "square_u32"),
    ("square", "f16", "square_f16"),
    ("sign", "f32", "sign_f32"),
    ("sign", "i32", "sign_i32"),
    ("sign", "u32", "sign_u32"),
    ("sign", "f16", "sign_f16"),
    // Float-only operations (only f32 and f16 entries)
    ("sqrt", "f32", "sqrt_f32"),
    ("sqrt", "f16", "sqrt_f16"),
    ("exp", "f32", "exp_f32"),
    ("exp", "f16", "exp_f16"),
    ("log", "f32", "log_f32"),
    ("log", "f16", "log_f16"),
    ("sin", "f32", "sin_f32"),
    ("sin", "f16", "sin_f16"),
    ("cos", "f32", "cos_f32"),
    ("cos", "f16", "cos_f16"),
    ("tan", "f32", "tan_f32"),
    ("tan", "f16", "tan_f16"),
    ("atan", "f32", "atan_f32"),
    ("atan", "f16", "atan_f16"),
    ("tanh", "f32", "tanh_f32"),
    ("tanh", "f16", "tanh_f16"),
    ("recip", "f32", "recip_f32"),
    ("recip", "f16", "recip_f16"),
    ("floor", "f32", "floor_f32"),
    ("floor", "f16", "floor_f16"),
    ("ceil", "f32", "ceil_f32"),
    ("ceil", "f16", "ceil_f16"),
    ("round", "f32", "round_f32"),
    ("round", "f16", "round_f16"),
    ("rsqrt", "f32", "rsqrt_f32"),
    ("rsqrt", "f16", "rsqrt_f16"),
    ("cbrt", "f32", "cbrt_f32"),
    ("cbrt", "f16", "cbrt_f16"),
    ("exp2", "f32", "exp2_f32"),
    ("exp2", "f16", "exp2_f16"),
    ("expm1", "f32", "expm1_f32"),
    ("expm1", "f16", "expm1_f16"),
    ("log2", "f32", "log2_f32"),
    ("log2", "f16", "log2_f16"),
    ("log10", "f32", "log10_f32"),
    ("log10", "f16", "log10_f16"),
    ("log1p", "f32", "log1p_f32"),
    ("log1p", "f16", "log1p_f16"),
    ("asin", "f32", "asin_f32"),
    ("asin", "f16", "asin_f16"),
    ("acos", "f32", "acos_f32"),
    ("acos", "f16", "acos_f16"),
    ("sinh", "f32", "sinh_f32"),
    ("sinh", "f16", "sinh_f16"),
    ("cosh", "f32", "cosh_f32"),
    ("cosh", "f16", "cosh_f16"),
    ("asinh", "f32", "asinh_f32"),
    ("asinh", "f16", "asinh_f16"),
    ("acosh", "f32", "acosh_f32"),
    ("acosh", "f16", "acosh_f16"),
    ("atanh", "f32", "atanh_f32"),
    ("atanh", "f16", "atanh_f16"),
    ("trunc", "f32", "trunc_f32"),
    ("trunc", "f16", "trunc_f16"),
];

/// Float-only operations that require float dtype
static FLOAT_ONLY_OPS: &[&str] = &[
    "sqrt", "rsqrt", "cbrt", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sin", "cos",
    "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "recip",
    "floor", "ceil", "round", "trunc",
];

/// Get the operation name string from UnaryOp enum
fn unary_op_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "neg",
        UnaryOp::Abs => "abs",
        UnaryOp::Square => "square",
        UnaryOp::Sign => "sign",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Rsqrt => "rsqrt",
        UnaryOp::Cbrt => "cbrt",
        UnaryOp::Exp => "exp",
        UnaryOp::Exp2 => "exp2",
        UnaryOp::Expm1 => "expm1",
        UnaryOp::Log => "log",
        UnaryOp::Log2 => "log2",
        UnaryOp::Log10 => "log10",
        UnaryOp::Log1p => "log1p",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tan => "tan",
        UnaryOp::Asin => "asin",
        UnaryOp::Acos => "acos",
        UnaryOp::Atan => "atan",
        UnaryOp::Sinh => "sinh",
        UnaryOp::Cosh => "cosh",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Asinh => "asinh",
        UnaryOp::Acosh => "acosh",
        UnaryOp::Atanh => "atanh",
        UnaryOp::Recip => "recip",
        UnaryOp::Floor => "floor",
        UnaryOp::Ceil => "ceil",
        UnaryOp::Round => "round",
        UnaryOp::Trunc => "trunc",
    }
}

/// Get the dtype from suffix string for error reporting
fn suffix_to_dtype(suffix: &str) -> DType {
    match suffix {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "i32" => DType::I32,
        "u32" => DType::U32,
        _ => DType::F32, // Fallback
    }
}

fn unary_op_to_entry_point(op: UnaryOp, suffix: &str) -> Result<&'static str> {
    let op_name = unary_op_name(op);
    let is_float = matches!(suffix, "f32" | "f16");

    // Check if this is a float-only operation being called with integer type
    if FLOAT_ONLY_OPS.contains(&op_name) && !is_float {
        return Err(Error::UnsupportedDType {
            dtype: suffix_to_dtype(suffix),
            op: match op {
                UnaryOp::Sqrt => "sqrt (requires float type)",
                UnaryOp::Exp => "exp (requires float type)",
                UnaryOp::Log => "log (requires float type)",
                UnaryOp::Sin => "sin (requires float type)",
                UnaryOp::Cos => "cos (requires float type)",
                UnaryOp::Tan => "tan (requires float type)",
                UnaryOp::Tanh => "tanh (requires float type)",
                UnaryOp::Recip => "recip (requires float type)",
                UnaryOp::Floor => "floor (requires float type)",
                UnaryOp::Ceil => "ceil (requires float type)",
                UnaryOp::Round => "round (requires float type)",
                _ => "unknown (requires float type)",
            },
        });
    }

    // Look up entry point in table
    for &(name, sfx, entry) in UNARY_ENTRY_POINTS {
        if name == op_name && sfx == suffix {
            return Ok(entry);
        }
    }

    // Fallback to f32 variant if exact match not found
    for &(name, sfx, entry) in UNARY_ENTRY_POINTS {
        if name == op_name && sfx == "f32" {
            return Ok(entry);
        }
    }

    // Should never reach here if lookup table is complete
    Err(Error::Internal(format!(
        "No entry point for {}_{}",
        op_name, suffix
    )))
}

/// Lookup table for scalar operation entry points: (op_name, suffix, entry_point)
static SCALAR_ENTRY_POINTS: &[(&str, &str, &str)] = &[
    // Add
    ("add", "f32", "add_scalar_f32"),
    ("add", "i32", "add_scalar_i32"),
    ("add", "u32", "add_scalar_u32"),
    ("add", "f16", "add_scalar_f16"),
    // Sub
    ("sub", "f32", "sub_scalar_f32"),
    ("sub", "i32", "sub_scalar_i32"),
    ("sub", "u32", "sub_scalar_u32"),
    ("sub", "f16", "sub_scalar_f16"),
    // Mul
    ("mul", "f32", "mul_scalar_f32"),
    ("mul", "i32", "mul_scalar_i32"),
    ("mul", "u32", "mul_scalar_u32"),
    ("mul", "f16", "mul_scalar_f16"),
    // Div
    ("div", "f32", "div_scalar_f32"),
    ("div", "i32", "div_scalar_i32"),
    ("div", "u32", "div_scalar_u32"),
    ("div", "f16", "div_scalar_f16"),
    // Pow
    ("pow", "f32", "pow_scalar_f32"),
    ("pow", "i32", "pow_scalar_i32"),
    ("pow", "u32", "pow_scalar_u32"),
    ("pow", "f16", "pow_scalar_f16"),
];

fn scalar_op_to_entry_point(op: BinaryOp, suffix: &str) -> Result<&'static str> {
    let op_name = binary_op_name(op);

    for &(name, sfx, entry) in SCALAR_ENTRY_POINTS {
        if name == op_name && sfx == suffix {
            return Ok(entry);
        }
    }

    Err(Error::Internal(format!(
        "No scalar entry point for {}_scalar_{}",
        op_name, suffix
    )))
}

/// Lookup table for reduce operation entry points: (op_name, suffix, entry_point)
static REDUCE_ENTRY_POINTS: &[(&str, &str, &str)] = &[
    // Sum
    ("sum", "f32", "reduce_sum_f32"),
    ("sum", "i32", "reduce_sum_i32"),
    ("sum", "u32", "reduce_sum_u32"),
    ("sum", "f16", "reduce_sum_f16"),
    // Max
    ("max", "f32", "reduce_max_f32"),
    ("max", "i32", "reduce_max_i32"),
    ("max", "u32", "reduce_max_u32"),
    ("max", "f16", "reduce_max_f16"),
    // Min
    ("min", "f32", "reduce_min_f32"),
    ("min", "i32", "reduce_min_i32"),
    ("min", "u32", "reduce_min_u32"),
    ("min", "f16", "reduce_min_f16"),
];

/// Get the kernel operation name for a reduce op.
/// Mean/Prod use Sum kernel (post-processing in Rust).
/// All/Any use Min kernel (logical reduction).
fn reduce_kernel_op(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Mean | ReduceOp::Prod => "sum", // Use sum kernel, post-process in Rust
        ReduceOp::All | ReduceOp::Any => "min",   // Use min kernel for logical reduction
    }
}

fn reduce_op_to_entry_point(op: ReduceOp, suffix: &str) -> Result<&'static str> {
    let kernel_op = reduce_kernel_op(op);

    for &(name, sfx, entry) in REDUCE_ENTRY_POINTS {
        if name == kernel_op && sfx == suffix {
            return Ok(entry);
        }
    }

    Err(Error::Internal(format!(
        "No reduce entry point for reduce_{}_{}",
        kernel_op, suffix
    )))
}

/// Lookup table for compare operation entry points: (op_name, suffix, entry_point)
static COMPARE_ENTRY_POINTS: &[(&str, &str, &str)] = &[
    // Eq
    ("eq", "f32", "eq_f32"),
    ("eq", "i32", "eq_i32"),
    ("eq", "u32", "eq_u32"),
    ("eq", "f16", "eq_f16"),
    // Ne
    ("ne", "f32", "ne_f32"),
    ("ne", "i32", "ne_i32"),
    ("ne", "u32", "ne_u32"),
    ("ne", "f16", "ne_f16"),
    // Lt
    ("lt", "f32", "lt_f32"),
    ("lt", "i32", "lt_i32"),
    ("lt", "u32", "lt_u32"),
    ("lt", "f16", "lt_f16"),
    // Le
    ("le", "f32", "le_f32"),
    ("le", "i32", "le_i32"),
    ("le", "u32", "le_u32"),
    ("le", "f16", "le_f16"),
    // Gt
    ("gt", "f32", "gt_f32"),
    ("gt", "i32", "gt_i32"),
    ("gt", "u32", "gt_u32"),
    ("gt", "f16", "gt_f16"),
    // Ge
    ("ge", "f32", "ge_f32"),
    ("ge", "i32", "ge_i32"),
    ("ge", "u32", "ge_u32"),
    ("ge", "f16", "ge_f16"),
];

fn compare_op_name(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "eq",
        CompareOp::Ne => "ne",
        CompareOp::Lt => "lt",
        CompareOp::Le => "le",
        CompareOp::Gt => "gt",
        CompareOp::Ge => "ge",
    }
}

fn compare_op_to_entry_point(op: CompareOp, suffix: &str) -> Result<&'static str> {
    let op_name = compare_op_name(op);

    for &(name, sfx, entry) in COMPARE_ENTRY_POINTS {
        if name == op_name && sfx == suffix {
            return Ok(entry);
        }
    }

    Err(Error::Internal(format!(
        "No compare entry point for {}_{}",
        op_name, suffix
    )))
}

// ============================================================================
// Uniform Buffer Structures
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryParams {
    numel: u32,
    _pad: [u32; 3], // Align to 16 bytes
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnaryParams {
    numel: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarParams {
    numel: u32,
    scalar: f32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    reduce_size: u32,
    outer_size: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    k: u32,
    n: u32,
    batch_size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormParams {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FillParams {
    numel: u32,
    value: f32,
    _pad: [u32; 2],
}

// ============================================================================
// TypedKernel Implementation Macro
// ============================================================================

/// Macro to implement TypedKernel for a specific dtype
macro_rules! impl_typed_kernel {
    ($rust_type:ty, $dtype:expr) => {
        impl TypedKernel<$rust_type> for WgpuClient {
            fn binary_op(&self, op: BinaryOp, a: u64, b: u64, out: u64, len: usize) -> Result<()> {
                let dtype = $dtype;
                let suffix = dtype_suffix(dtype)?;

                let cache = shader_cache();
                let module = cache.get_or_create_module(self, dtype, "binary", || {
                    generate_binary_shader(dtype)
                })?;

                let entry_point = binary_op_to_entry_point(op, suffix)?;
                let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
                    num_storage_buffers: 3,
                    num_uniform_buffers: 1,
                });
                let pipeline = cache.get_or_create_pipeline(
                    self,
                    dtype,
                    "binary",
                    entry_point,
                    &module,
                    &layout,
                );

                let a_buf =
                    get_buffer(a).ok_or_else(|| Error::Internal("Invalid buffer A".into()))?;
                let b_buf =
                    get_buffer(b).ok_or_else(|| Error::Internal("Invalid buffer B".into()))?;
                let out_buf =
                    get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

                let params = BinaryParams {
                    numel: len as u32,
                    _pad: [0; 3],
                };
                let params_buf = self.create_uniform_buffer("binary_params", 16);
                self.write_buffer(&params_buf, &[params]);

                let bind_group = self
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("binary_bind_group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("binary_op"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("binary_op"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    pass.dispatch_workgroups(workgroup_count(len), 1, 1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }

            fn unary_op(&self, op: UnaryOp, a: u64, out: u64, len: usize) -> Result<()> {
                let dtype = $dtype;
                let suffix = dtype_suffix(dtype)?;

                let cache = shader_cache();
                let module = cache
                    .get_or_create_module(self, dtype, "unary", || generate_unary_shader(dtype))?;

                let entry_point = unary_op_to_entry_point(op, suffix)?;
                let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
                    num_storage_buffers: 2,
                    num_uniform_buffers: 1,
                });
                let pipeline = cache.get_or_create_pipeline(
                    self,
                    dtype,
                    "unary",
                    entry_point,
                    &module,
                    &layout,
                );

                let a_buf =
                    get_buffer(a).ok_or_else(|| Error::Internal("Invalid buffer A".into()))?;
                let out_buf =
                    get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

                let params = UnaryParams {
                    numel: len as u32,
                    _pad: [0; 3],
                };
                let params_buf = self.create_uniform_buffer("unary_params", 16);
                self.write_buffer(&params_buf, &[params]);

                let bind_group = self
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("unary_bind_group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("unary_op"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("unary_op"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    pass.dispatch_workgroups(workgroup_count(len), 1, 1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }

            fn scalar_op(
                &self,
                op: BinaryOp,
                a: u64,
                scalar: f64,
                out: u64,
                len: usize,
            ) -> Result<()> {
                let dtype = $dtype;
                let suffix = dtype_suffix(dtype)?;

                let cache = shader_cache();
                let module = cache.get_or_create_module(self, dtype, "scalar", || {
                    generate_scalar_shader(dtype)
                })?;

                let entry_point = scalar_op_to_entry_point(op, suffix)?;
                let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
                    num_storage_buffers: 2,
                    num_uniform_buffers: 1,
                });
                let pipeline = cache.get_or_create_pipeline(
                    self,
                    dtype,
                    "scalar",
                    entry_point,
                    &module,
                    &layout,
                );

                let a_buf =
                    get_buffer(a).ok_or_else(|| Error::Internal("Invalid buffer A".into()))?;
                let out_buf =
                    get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

                let params = ScalarParams {
                    numel: len as u32,
                    scalar: scalar as f32,
                    _pad: [0; 2],
                };
                let params_buf = self.create_uniform_buffer("scalar_params", 16);
                self.write_buffer(&params_buf, &[params]);

                let bind_group = self
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("scalar_bind_group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("scalar_op"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("scalar_op"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    pass.dispatch_workgroups(workgroup_count(len), 1, 1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }

            fn reduce(
                &self,
                op: ReduceOp,
                a: u64,
                out: u64,
                reduce_size: usize,
                outer_size: usize,
            ) -> Result<()> {
                let dtype = $dtype;
                let suffix = dtype_suffix(dtype)?;

                let cache = shader_cache();
                let module = cache.get_or_create_module(self, dtype, "reduce", || {
                    generate_reduce_shader(dtype)
                })?;

                let entry_point = reduce_op_to_entry_point(op, suffix)?;
                let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
                    num_storage_buffers: 2,
                    num_uniform_buffers: 1,
                });
                let pipeline = cache.get_or_create_pipeline(
                    self,
                    dtype,
                    "reduce",
                    entry_point,
                    &module,
                    &layout,
                );

                let a_buf =
                    get_buffer(a).ok_or_else(|| Error::Internal("Invalid buffer A".into()))?;
                let out_buf =
                    get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

                let params = ReduceParams {
                    reduce_size: reduce_size as u32,
                    outer_size: outer_size as u32,
                    _pad: [0; 2],
                };
                let params_buf = self.create_uniform_buffer("reduce_params", 16);
                self.write_buffer(&params_buf, &[params]);

                let bind_group = self
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("reduce_bind_group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("reduce_op"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("reduce_op"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    pass.dispatch_workgroups(outer_size as u32, 1, 1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }

            fn fill(&self, out: u64, value: f64, len: usize) -> Result<()> {
                let dtype = $dtype;

                let cache = shader_cache();
                let module = cache
                    .get_or_create_module(self, dtype, "fill", || generate_fill_shader(dtype))?;

                let suffix = dtype_suffix(dtype)?;
                let entry_point = match suffix {
                    "f32" => "fill_f32",
                    "i32" => "fill_i32",
                    "u32" => "fill_u32",
                    _ => "fill_f32",
                };

                let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
                    num_storage_buffers: 1,
                    num_uniform_buffers: 1,
                });
                let pipeline = cache.get_or_create_pipeline(
                    self,
                    dtype,
                    "fill",
                    entry_point,
                    &module,
                    &layout,
                );

                let out_buf =
                    get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

                let params = FillParams {
                    numel: len as u32,
                    value: value as f32,
                    _pad: [0; 2],
                };
                let params_buf = self.create_uniform_buffer("fill_params", 16);
                self.write_buffer(&params_buf, &[params]);

                let bind_group = self
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("fill_bind_group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("fill_op"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("fill_op"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    pass.dispatch_workgroups(workgroup_count(len), 1, 1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }

            fn copy(&self, src: u64, dst: u64, len: usize) -> Result<()> {
                let dtype = $dtype;
                let elem_size = dtype.size_in_bytes();
                let size_bytes = (len * elem_size) as u64;

                let src_buf =
                    get_buffer(src).ok_or_else(|| Error::Internal("Invalid buffer src".into()))?;
                let dst_buf =
                    get_buffer(dst).ok_or_else(|| Error::Internal("Invalid buffer dst".into()))?;

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("copy"),
                        });
                encoder.copy_buffer_to_buffer(&src_buf, 0, &dst_buf, 0, size_bytes);
                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }
        }

        impl TypedMatmul<$rust_type> for WgpuClient {
            fn matmul(
                &self,
                a: u64,
                b: u64,
                out: u64,
                m: usize,
                n: usize,
                k: usize,
                _lda: usize,
                _ldb: usize,
                _ldc: usize,
            ) -> Result<()> {
                let dtype = $dtype;

                let cache = shader_cache();
                let module = cache.get_or_create_module(self, dtype, "matmul", || {
                    generate_matmul_shader(dtype)
                })?;

                let suffix = dtype_suffix(dtype)?;
                let entry_point = match suffix {
                    "f32" => "matmul_f32",
                    "i32" => "matmul_i32",
                    "u32" => "matmul_u32",
                    _ => "matmul_f32",
                };

                let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
                    num_storage_buffers: 3,
                    num_uniform_buffers: 1,
                });
                let pipeline = cache.get_or_create_pipeline(
                    self,
                    dtype,
                    "matmul",
                    entry_point,
                    &module,
                    &layout,
                );

                let a_buf =
                    get_buffer(a).ok_or_else(|| Error::Internal("Invalid buffer A".into()))?;
                let b_buf =
                    get_buffer(b).ok_or_else(|| Error::Internal("Invalid buffer B".into()))?;
                let out_buf =
                    get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

                let params = MatmulParams {
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    batch_size: 1,
                };
                let params_buf = self.create_uniform_buffer("matmul_params", 16);
                self.write_buffer(&params_buf, &[params]);

                let bind_group = self
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("matmul_bind_group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("matmul"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("matmul"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    // Dispatch with tile size 16x16
                    let tile_size = 16u32;
                    let workgroups_x = (n as u32 + tile_size - 1) / tile_size;
                    let workgroups_y = (m as u32 + tile_size - 1) / tile_size;
                    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }

            fn batched_matmul(
                &self,
                a: u64,
                b: u64,
                out: u64,
                batch_size: usize,
                m: usize,
                n: usize,
                k: usize,
            ) -> Result<()> {
                let dtype = $dtype;

                let cache = shader_cache();
                let module = cache.get_or_create_module(self, dtype, "matmul", || {
                    generate_matmul_shader(dtype)
                })?;

                let suffix = dtype_suffix(dtype)?;
                let entry_point = match suffix {
                    "f32" => "batched_matmul_f32",
                    "i32" => "batched_matmul_i32",
                    "u32" => "batched_matmul_u32",
                    _ => "batched_matmul_f32",
                };

                let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
                    num_storage_buffers: 3,
                    num_uniform_buffers: 1,
                });
                let pipeline = cache.get_or_create_pipeline(
                    self,
                    dtype,
                    "matmul",
                    entry_point,
                    &module,
                    &layout,
                );

                let a_buf =
                    get_buffer(a).ok_or_else(|| Error::Internal("Invalid buffer A".into()))?;
                let b_buf =
                    get_buffer(b).ok_or_else(|| Error::Internal("Invalid buffer B".into()))?;
                let out_buf =
                    get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

                let params = MatmulParams {
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    batch_size: batch_size as u32,
                };
                let params_buf = self.create_uniform_buffer("matmul_params", 16);
                self.write_buffer(&params_buf, &[params]);

                let bind_group = self
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("batched_matmul_bind_group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("batched_matmul"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("batched_matmul"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    let tile_size = 16u32;
                    let workgroups_x = (n as u32 + tile_size - 1) / tile_size;
                    let workgroups_y = (m as u32 + tile_size - 1) / tile_size;
                    pass.dispatch_workgroups(workgroups_x, workgroups_y, batch_size as u32);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }
        }

        impl TypedCompare<$rust_type> for WgpuClient {
            fn compare(&self, op: CompareOp, a: u64, b: u64, out: u64, len: usize) -> Result<()> {
                let dtype = $dtype;
                let suffix = dtype_suffix(dtype)?;

                let cache = shader_cache();
                let module = cache.get_or_create_module(self, dtype, "compare", || {
                    generate_compare_shader(dtype)
                })?;

                let entry_point = compare_op_to_entry_point(op, suffix)?;
                let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
                    num_storage_buffers: 3,
                    num_uniform_buffers: 1,
                });
                let pipeline = cache.get_or_create_pipeline(
                    self,
                    dtype,
                    "compare",
                    entry_point,
                    &module,
                    &layout,
                );

                let a_buf =
                    get_buffer(a).ok_or_else(|| Error::Internal("Invalid buffer A".into()))?;
                let b_buf =
                    get_buffer(b).ok_or_else(|| Error::Internal("Invalid buffer B".into()))?;
                let out_buf =
                    get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

                let params = BinaryParams {
                    numel: len as u32,
                    _pad: [0; 3],
                };
                let params_buf = self.create_uniform_buffer("compare_params", 16);
                self.write_buffer(&params_buf, &[params]);

                let bind_group = self
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("compare_bind_group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    self.wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("compare_op"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("compare_op"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    pass.dispatch_workgroups(workgroup_count(len), 1, 1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                Ok(())
            }
        }
    };
}

// ============================================================================
// TypedNorm Implementation (Float types only)
// ============================================================================

impl TypedNorm<f32> for WgpuClient {
    fn rms_norm(
        &self,
        input: u64,
        weight: u64,
        out: u64,
        batch_size: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        let dtype = DType::F32;

        let cache = shader_cache();
        let module =
            cache.get_or_create_module(self, dtype, "norm", || generate_norm_shader(dtype))?;

        let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 3,
            num_uniform_buffers: 1,
        });
        let pipeline =
            cache.get_or_create_pipeline(self, dtype, "norm", "rms_norm_f32", &module, &layout);

        let input_buf =
            get_buffer(input).ok_or_else(|| Error::Internal("Invalid buffer input".into()))?;
        let weight_buf =
            get_buffer(weight).ok_or_else(|| Error::Internal("Invalid buffer weight".into()))?;
        let out_buf =
            get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

        let params = NormParams {
            batch_size: batch_size as u32,
            hidden_size: hidden_size as u32,
            eps,
            _pad: 0,
        };
        let params_buf = self.create_uniform_buffer("rms_norm_params", 16);
        self.write_buffer(&params_buf, &[params]);

        let bind_group = self
            .wgpu_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rms_norm_bind_group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rms_norm"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rms_norm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn layer_norm(
        &self,
        input: u64,
        weight: u64,
        bias: u64,
        out: u64,
        batch_size: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        let dtype = DType::F32;

        let cache = shader_cache();
        let module =
            cache.get_or_create_module(self, dtype, "norm", || generate_norm_shader(dtype))?;

        let layout = self.pipeline_cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 4,
            num_uniform_buffers: 1,
        });
        let pipeline =
            cache.get_or_create_pipeline(self, dtype, "norm", "layer_norm_f32", &module, &layout);

        let input_buf =
            get_buffer(input).ok_or_else(|| Error::Internal("Invalid buffer input".into()))?;
        let weight_buf =
            get_buffer(weight).ok_or_else(|| Error::Internal("Invalid buffer weight".into()))?;
        let bias_buf =
            get_buffer(bias).ok_or_else(|| Error::Internal("Invalid buffer bias".into()))?;
        let out_buf =
            get_buffer(out).ok_or_else(|| Error::Internal("Invalid buffer out".into()))?;

        let params = NormParams {
            batch_size: batch_size as u32,
            hidden_size: hidden_size as u32,
            eps,
            _pad: 0,
        };
        let params_buf = self.create_uniform_buffer("layer_norm_params", 16);
        self.write_buffer(&params_buf, &[params]);

        let bind_group = self
            .wgpu_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layer_norm_bind_group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bias_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("layer_norm"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer_norm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

// Stub implementations for integer TypedNorm (normalization is float-only)
impl TypedNorm<i32> for WgpuClient {
    fn rms_norm(&self, _: u64, _: u64, _: u64, _: usize, _: usize, _: f32) -> Result<()> {
        Err(Error::UnsupportedDType {
            dtype: DType::I32,
            op: "rms_norm (requires float type)",
        })
    }
    fn layer_norm(&self, _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: f32) -> Result<()> {
        Err(Error::UnsupportedDType {
            dtype: DType::I32,
            op: "layer_norm (requires float type)",
        })
    }
}

impl TypedNorm<u32> for WgpuClient {
    fn rms_norm(&self, _: u64, _: u64, _: u64, _: usize, _: usize, _: f32) -> Result<()> {
        Err(Error::UnsupportedDType {
            dtype: DType::U32,
            op: "rms_norm (requires float type)",
        })
    }
    fn layer_norm(&self, _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: f32) -> Result<()> {
        Err(Error::UnsupportedDType {
            dtype: DType::U32,
            op: "layer_norm (requires float type)",
        })
    }
}

// ============================================================================
// Implement TypedKernel for each dtype
// ============================================================================

impl_typed_kernel!(f32, DType::F32);
impl_typed_kernel!(i32, DType::I32);
impl_typed_kernel!(u32, DType::U32);

// ============================================================================
// Compile-Time Enforcement: WgpuKernels Marker Trait
// ============================================================================

/// **CRITICAL**: This line enforces compile-time dtype coverage.
///
/// If ANY of the following are missing, this line will cause a compile error:
/// - `TypedKernel<f32> for WgpuClient`
/// - `TypedKernel<i32> for WgpuClient`
/// - `TypedKernel<u32> for WgpuClient`
/// - `TypedMatmul<f32> for WgpuClient`
/// - `TypedMatmul<i32> for WgpuClient`
/// - `TypedMatmul<u32> for WgpuClient`
/// - `TypedNorm<f32> for WgpuClient`
/// - `TypedCompare<f32> for WgpuClient`
/// - `TypedCompare<i32> for WgpuClient`
/// - `TypedCompare<u32> for WgpuClient`
impl WgpuKernels for WgpuClient {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_kernels_trait_enforced() {
        // This test verifies that WgpuKernels is implemented.
        // If any TypedKernel<T> is missing, this won't compile.
        fn assert_wgpu_kernels<T: WgpuKernels>() {}
        assert_wgpu_kernels::<WgpuClient>();
    }

    #[test]
    fn test_shader_generation() {
        // Test that shaders generate without error
        let binary_f32 = generate_binary_shader(DType::F32).unwrap();
        assert!(binary_f32.contains("fn add_f32"));
        assert!(binary_f32.contains("array<f32>"));

        let binary_i32 = generate_binary_shader(DType::I32).unwrap();
        assert!(binary_i32.contains("fn add_i32"));
        assert!(binary_i32.contains("array<i32>"));

        let binary_u32 = generate_binary_shader(DType::U32).unwrap();
        assert!(binary_u32.contains("fn add_u32"));
        assert!(binary_u32.contains("array<u32>"));
    }
}
