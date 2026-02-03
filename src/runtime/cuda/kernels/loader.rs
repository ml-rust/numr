//! CUDA kernel loading, caching, and launching infrastructure
//!
//! This module provides utilities for loading PTX kernels compiled by build.rs,
//! caching the modules per-device, and launching kernels with type-safe wrappers.
//!
//! # Architecture
//!
//! - PTX files are compiled by `build.rs` using nvcc
//! - Modules are loaded on first use and cached per-device
//! - Generic launch helpers reduce boilerplate across kernel types
//!
//! # Thread Safety
//!
//! The module cache uses `OnceLock<Mutex<HashMap>>` for thread-safe initialization
//! and concurrent access from multiple CUDA streams.

use cudarc::driver::PushKernelArg;
pub use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// PTX Sources (compiled by build.rs)
// ============================================================================

/// Directory containing compiled PTX files (set by build.rs)
const KERNEL_DIR: &str = env!("CUDA_KERNEL_DIR");

/// Load PTX from compiled file.
fn load_ptx(name: &str) -> Ptx {
    let path = format!("{}/{}.ptx", KERNEL_DIR, name);
    Ptx::from_file(path)
}

// ============================================================================
// Kernel Module Cache
// ============================================================================

/// Cache for loaded CUDA modules, keyed by (device_index, module_name)
static MODULE_CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), Arc<CudaModule>>>> =
    OnceLock::new();

/// Get or load a CUDA module from PTX.
///
/// Modules are cached per-device to avoid repeated loading. This is thread-safe
/// and can be called concurrently from multiple streams.
///
/// # Arguments
///
/// * `context` - CUDA context for the target device
/// * `device_index` - Index of the target device (used as cache key)
/// * `module_name` - Name of the PTX file (without extension)
///
/// # Errors
///
/// Returns an error if the PTX file cannot be loaded or the module cannot be created.
pub fn get_or_load_module(
    context: &Arc<CudaContext>,
    device_index: usize,
    module_name: &'static str,
) -> Result<Arc<CudaModule>> {
    let cache = MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().map_err(|e| {
        Error::Internal(format!(
            "Failed to acquire module cache lock (Mutex poisoned): {}",
            e
        ))
    })?;

    let key = (device_index, module_name);
    if let Some(module) = guard.get(&key) {
        return Ok(module.clone());
    }

    // Load PTX and create module
    let ptx = load_ptx(module_name);
    let module = context.load_module(ptx).map_err(|e| {
        Error::Internal(format!(
            "Failed to load CUDA module '{}': {:?}. \
             Ensure CUDA kernels were compiled correctly by build.rs.",
            module_name, e
        ))
    })?;

    guard.insert(key, module.clone());

    Ok(module)
}

/// Get a kernel function from a loaded module.
///
/// # Arguments
///
/// * `module` - Loaded CUDA module
/// * `kernel_name` - Name of the kernel function (e.g., "add_f32")
///
/// # Errors
///
/// Returns an error if the kernel function is not found in the module.
pub fn get_kernel_function(module: &Arc<CudaModule>, kernel_name: &str) -> Result<CudaFunction> {
    module.load_function(kernel_name).map_err(|e| {
        Error::Internal(format!(
            "Failed to get kernel '{}': {:?}. \
             Check that the kernel name matches the CUDA source.",
            kernel_name, e
        ))
    })
}

// ============================================================================
// Launch Configuration
// ============================================================================

/// Block size for element-wise operations (256 threads is optimal for most GPUs)
pub const BLOCK_SIZE: u32 = 256;

/// Calculate optimal grid dimensions for element-wise operations.
///
/// Uses a 1D grid with blocks of `BLOCK_SIZE` threads each.
#[inline]
pub fn elementwise_launch_config(numel: usize) -> (u32, u32, u32) {
    let grid_size = ((numel as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    (grid_size, 1, 1)
}

/// Calculate launch configuration for global reduction kernels.
///
/// Limits grid size to prevent excessive block overhead for small inputs.
#[inline]
#[allow(dead_code)] // Kept for potential future optimization of global reductions
pub fn reduce_launch_config(numel: usize) -> (u32, u32) {
    let block_size = BLOCK_SIZE;
    let grid_size = ((numel as u32) + block_size - 1) / block_size;
    // Limit grid size to ensure we don't launch too many blocks
    let grid_size = grid_size.min(1024);
    (grid_size, block_size)
}

/// Calculate launch configuration for dimension-wise reduction.
///
/// Uses a 2D grid where each (outer, inner) pair is processed by one thread block.
#[inline]
pub fn reduce_dim_launch_config(outer: usize, inner: usize) -> ((u32, u32, u32), u32) {
    let grid = (outer as u32, inner as u32, 1);
    let block = BLOCK_SIZE;
    (grid, block)
}

/// Calculate launch configuration for softmax over the last dimension.
///
/// One block per row, with threads cooperating to compute the softmax.
/// Returns (grid_size, block_size, shared_memory_bytes).
#[inline]
pub fn softmax_launch_config(outer: usize, dim_size: usize) -> (u32, u32, u32) {
    // One block per row, threads handle the dimension
    let block_size = BLOCK_SIZE.min(dim_size as u32);
    let grid_size = outer as u32;
    // Shared memory: 2 arrays of block_size floats (for max and sum reduction)
    let shared_mem = 2 * block_size * 4; // f32
    (grid_size, block_size, shared_mem)
}

/// Calculate launch configuration for softmax over a non-last dimension.
///
/// Uses a 2D grid to process all (outer, inner) pairs in parallel.
/// Each thread processes one element position across the reduction dimension.
#[inline]
#[allow(dead_code)] // Available for future optimized softmax_dim kernel
pub fn softmax_dim_launch_config(outer: usize, inner: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    // Use 2D grid: one thread per (outer, inner) pair
    // Each thread sequentially processes the dim_size elements
    let total_elements = (outer * inner) as u32;
    let grid_x = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, 1, 1);
    let block = (BLOCK_SIZE, 1, 1);
    (grid, block)
}

/// Create a launch configuration from grid, block, and shared memory sizes.
#[inline]
pub fn launch_config(
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: shared_mem,
    }
}

// ============================================================================
// Kernel Naming
// ============================================================================

/// Kernel operation categories for consistent naming.
pub mod kernel_names {
    /// Binary operations (two tensor inputs)
    pub const BINARY_MODULE: &str = "binary";
    /// Unary operations (one tensor input)
    pub const UNARY_MODULE: &str = "unary";
    /// Scalar operations (tensor + scalar input)
    pub const SCALAR_MODULE: &str = "scalar";
    /// Reduction operations (sum, max, min)
    pub const REDUCE_MODULE: &str = "reduce";
    /// Comparison operations (eq, ne, lt, le, gt, ge)
    pub const COMPARE_MODULE: &str = "compare";
    /// Activation functions (relu, sigmoid, softmax, silu, gelu)
    pub const ACTIVATION_MODULE: &str = "activation";
    /// Normalization operations (rms_norm, layer_norm)
    pub const NORM_MODULE: &str = "norm";
    /// Type casting operations (cast between dtypes)
    pub const CAST_MODULE: &str = "cast";
    /// Utility operations (fill)
    pub const UTILITY_MODULE: &str = "utility";
    /// Ternary operations (where)
    pub const TERNARY_MODULE: &str = "ternary";
    /// Prefix sum operations (exclusive scan)
    #[cfg(feature = "sparse")]
    pub const SCAN_MODULE: &str = "scan";
    /// Sparse matrix operations (SpMV, SpMM)
    #[cfg(feature = "sparse")]
    pub const SPARSE_SPMV_MODULE: &str = "sparse_spmv";
    /// Sparse matrix element-wise operations (add, sub, mul)
    #[cfg(feature = "sparse")]
    pub const SPARSE_MERGE_MODULE: &str = "sparse_merge";
    /// Sparse format conversion operations (COO↔CSR↔CSC)
    #[cfg(feature = "sparse")]
    pub const SPARSE_CONVERT_MODULE: &str = "sparse_convert";
    /// COO sparse element-wise operations with CUB sort
    #[cfg(feature = "sparse")]
    pub const SPARSE_COO_MODULE: &str = "sparse_coo";
    /// Dense × Sparse matrix multiplication (DSMM / SpMM)
    #[cfg(feature = "sparse")]
    pub const DSMM_MODULE: &str = "dsmm";
    /// Linear algebra basic operations (trace, diag, diagflat, identity, transpose)
    pub const LINALG_BASIC_MODULE: &str = "linalg_basic";
    /// Linear algebra solvers (forward_sub, backward_sub, det_from_lu, apply_permutation)
    pub const LINALG_SOLVERS_MODULE: &str = "linalg_solvers";
    /// Matrix decompositions (LU, Cholesky, QR)
    pub const LINALG_DECOMP_MODULE: &str = "linalg_decomp";
    /// SVD decomposition (Jacobi algorithm)
    pub const LINALG_SVD_MODULE: &str = "linalg_svd";
    /// Symmetric eigenvalue decomposition (Jacobi algorithm)
    pub const LINALG_EIGEN_MODULE: &str = "linalg_eigen";
    /// Schur decomposition (Hessenberg + QR iteration)
    pub const LINALG_SCHUR_MODULE: &str = "linalg_schur";
    /// General eigenvalue decomposition
    pub const LINALG_EIGEN_GENERAL_MODULE: &str = "linalg_eigen_general";
    /// Advanced decompositions (rsf2csf)
    pub const LINALG_ADVANCED_MODULE: &str = "linalg_advanced";
    /// QZ decomposition (generalized Schur - double-shift algorithm)
    pub const LINALG_QZ_MODULE: &str = "linalg_qz";
    /// Matrix functions (exp, log, sqrt on quasi-triangular matrices)
    pub const LINALG_MATRIX_FUNCS_MODULE: &str = "linalg_matrix_funcs";
    /// Matrix multiplication operations (native tiled GEMM)
    pub const MATMUL_MODULE: &str = "matmul";
    /// Cumulative operations (cumsum, cumprod, logsumexp)
    pub const CUMULATIVE_MODULE: &str = "cumulative";
    /// Distribution sampling operations (bernoulli, beta, gamma, etc.)
    pub const DISTRIBUTIONS_MODULE: &str = "distributions";
    /// Quasi-random sequence generation (sobol, halton, latin_hypercube)
    pub const QUASIRANDOM_MODULE: &str = "quasirandom";
    /// Advanced PRNGs (philox, threefry, pcg64, xoshiro256)
    pub const ADVANCED_RANDOM_MODULE: &str = "advanced_random";
    /// Statistics operations (mode)
    pub const STATISTICS_MODULE: &str = "statistics";

    /// Generate kernel name for reduction operations.
    #[inline]
    pub fn reduce_kernel(op: &str) -> String {
        format!("reduce_{}", op)
    }

    /// Generate kernel name for dimension-wise reduction operations.
    #[inline]
    pub fn reduce_dim_kernel(op: &str) -> String {
        format!("reduce_{}_dim", op)
    }
}

/// Get the kernel name suffix for a given dtype.
pub fn dtype_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::FP8E4M3 => "fp8_e4m3",
        DType::FP8E5M2 => "fp8_e5m2",
        DType::I64 => "i64",
        DType::I32 => "i32",
        DType::I16 => "i16",
        DType::I8 => "i8",
        DType::U64 => "u64",
        DType::U32 => "u32",
        DType::U16 => "u16",
        DType::U8 => "u8",
        DType::Bool => "bool",
        DType::Complex64 => "c64",
        DType::Complex128 => "c128",
    }
}

/// Generate a kernel name with dtype suffix.
///
/// # Example
///
/// ```ignore
/// let name = kernel_name("add", DType::F32); // "add_f32"
/// ```
#[inline]
pub fn kernel_name(base: &str, dtype: DType) -> String {
    format!("{}_{}", base, dtype_suffix(dtype))
}

// ============================================================================
// Generic Kernel Launch Helpers
// ============================================================================

/// Launch an element-wise unary kernel (one input, one output).
///
/// This handles the common pattern for operations like neg, abs, sqrt, exp, etc.
///
/// # Safety
///
/// `input_ptr` and `output_ptr` must be valid device memory pointers with at least
/// `numel` elements of the appropriate dtype.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `module_name` - PTX module name (e.g., "unary", "activation")
/// * `op` - Operation name (e.g., "neg", "relu")
/// * `dtype` - Data type of the tensors
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_unary_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    module_name: &'static str,
    op: &str,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(op, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel '{}' launch failed: {:?}",
                module_name, op, e
            ))
        })?;

        Ok(())
    }
}

/// Launch an element-wise binary kernel (two inputs, one output).
///
/// This handles the common pattern for operations like add, sub, mul, div, etc.
///
/// # Safety
///
/// All pointers must be valid device memory with at least `numel` elements.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `module_name` - PTX module name (e.g., "binary", "compare")
/// * `op` - Operation name (e.g., "add", "eq")
/// * `dtype` - Data type of the tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_binary_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    module_name: &'static str,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(op, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel '{}' launch failed: {:?}",
                module_name, op, e
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// Matrix Multiplication Launch Helpers
// ============================================================================

use crate::algorithm::TileConfig;

/// Calculate launch configuration for register-tiled matrix multiplication.
///
/// Uses configurable tile sizes - no hardcoded values.
/// Grid: ceil(N/block_n) × ceil(M/block_m)
/// Block: (block_n/thread_n) × (block_m/thread_m) threads
#[inline]
pub fn matmul_launch_config(
    m: usize,
    n: usize,
    cfg: &TileConfig,
    elem_size: usize,
) -> LaunchConfig {
    let grid_x = ((n as u32) + cfg.block_n as u32 - 1) / cfg.block_n as u32;
    let grid_y = ((m as u32) + cfg.block_m as u32 - 1) / cfg.block_m as u32;
    let threads_x = cfg.block_n / cfg.thread_n;
    let threads_y = cfg.block_m / cfg.thread_m;

    // Dynamic shared memory: As[block_m][block_k] + Bs[block_k][block_n]
    let shared_mem_bytes = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * elem_size;

    LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (threads_x as u32, threads_y as u32, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    }
}

/// Calculate launch configuration for batched register-tiled matrix multiplication.
///
/// Uses 3D grid: (tiles_x, tiles_y, batch)
#[inline]
pub fn matmul_batched_launch_config(
    batch: usize,
    m: usize,
    n: usize,
    cfg: &TileConfig,
    elem_size: usize,
) -> LaunchConfig {
    let grid_x = ((n as u32) + cfg.block_n as u32 - 1) / cfg.block_n as u32;
    let grid_y = ((m as u32) + cfg.block_m as u32 - 1) / cfg.block_m as u32;
    let grid_z = batch as u32;
    let threads_x = cfg.block_n / cfg.thread_n;
    let threads_y = cfg.block_m / cfg.thread_m;

    let shared_mem_bytes = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * elem_size;

    LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (threads_x as u32, threads_y as u32, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    }
}

/// Get default tile configuration for a dtype.
///
/// These are reasonable defaults; can be overridden via autotuning.
#[inline]
pub fn default_tile_config(dtype: DType) -> TileConfig {
    match dtype {
        // F64 uses smaller tiles due to larger element size
        DType::F64 => TileConfig {
            block_m: 64,
            block_n: 64,
            block_k: 8,
            thread_m: 4,
            thread_n: 4,
        },
        // F32/F16/BF16 use larger tiles
        _ => TileConfig::CUDA,
    }
}

/// Launch native tiled matmul kernel: C[M,N] = A[M,K] @ B[K,N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: M * K elements
/// - B: K * N elements
/// - C: M * N elements
pub unsafe fn launch_matmul_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    unsafe {
        launch_matmul_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            &default_tile_config(dtype),
        )
    }
}

/// Launch native tiled matmul kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_kernel_with_config(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    // For F16/BF16, shared memory uses F32 for accumulation
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4, // F32 accumulator
        _ => elem_size,
    };

    let cfg = matmul_launch_config(m, n, tile_cfg, shared_elem_size);
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA matmul kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch native batched tiled matmul kernel: C[batch,M,N] = A[batch,M,K] @ B[batch,K,N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: batch * M * K elements
/// - B: batch * K * N elements
/// - C: batch * M * N elements
pub unsafe fn launch_matmul_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    unsafe {
        launch_matmul_batched_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            c_ptr,
            batch,
            m,
            n,
            k,
            &default_tile_config(dtype),
        )
    }
}

/// Launch native batched tiled matmul kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_batched_kernel_with_config(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };

    let cfg = matmul_batched_launch_config(batch, m, n, tile_cfg, shared_elem_size);
    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA batched matmul kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

// ============================================================================
// Fused Matmul+Bias Kernel Launch
// ============================================================================

/// Launch native tiled fused matmul+bias kernel: C[M,N] = A[M,K] @ B[K,N] + bias[N]
///
/// Uses the same tiled GEMM algorithm as matmul, but fuses bias addition into the
/// epilogue to avoid an extra memory round-trip.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: M * K elements
/// - B: K * N elements
/// - bias: N elements (1D, broadcast across rows)
/// - C: M * N elements (output)
pub unsafe fn launch_matmul_bias_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    unsafe {
        launch_matmul_bias_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            bias_ptr,
            c_ptr,
            m,
            n,
            k,
            &default_tile_config(dtype),
        )
    }
}

/// Launch native tiled fused matmul+bias kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_bias_kernel_with_config(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_bias", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    // For F16/BF16, shared memory uses F32 for accumulation
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4, // F32 accumulator
        _ => elem_size,
    };

    let cfg = matmul_launch_config(m, n, tile_cfg, shared_elem_size);
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA matmul_bias kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch native batched tiled fused matmul+bias kernel:
/// C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] + bias[N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: batch * M * K elements
/// - B: batch * K * N elements
/// - bias: N elements (1D, broadcast across all batches and rows)
/// - C: batch * M * N elements (output)
pub unsafe fn launch_matmul_bias_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    unsafe {
        launch_matmul_bias_batched_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            bias_ptr,
            c_ptr,
            batch,
            m,
            n,
            k,
            &default_tile_config(dtype),
        )
    }
}

/// Launch native batched tiled fused matmul+bias kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_bias_batched_kernel_with_config(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_bias_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };

    let cfg = matmul_batched_launch_config(batch, m, n, tile_cfg, shared_elem_size);
    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA batched matmul_bias kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
