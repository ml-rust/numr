//! WGSL kernel launchers for sparse matrix-vector and matrix-matrix multiplication.
//!
//! Provides launchers for CSR format SpMV and SpMM operations:
//! - `launch_csr_spmv` - Sparse matrix-vector multiplication: y = A * x
//! - `launch_csr_spmm` - Sparse matrix-dense matrix multiplication: C = A * B

use wgpu::{Buffer, Queue};

use super::generator::dtype_suffix;
use super::generator::spmv::{
    generate_csr_extract_diagonal_shader, generate_csr_spmm_shader, generate_csr_spmv_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

/// Launch CSR SpMV kernel: y = A * x
///
/// Row-parallel implementation where each thread processes one row.
///
/// # Buffers
///
/// - `row_ptrs`: CSR row pointers [nrows + 1] (I32)
/// - `col_indices`: CSR column indices [nnz] (I32)
/// - `values`: CSR values [nnz] (dtype)
/// - `x`: Dense input vector [ncols] (dtype)
/// - `y`: Dense output vector [nrows] (dtype)
/// - `params`: Uniform buffer with SpmvParams { nrows, ncols }
pub fn launch_csr_spmv(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    values: &Buffer,
    x: &Buffer,
    y: &Buffer,
    params_buffer: &Buffer,
    nrows: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csr_spmv_{}", suffix);

    let shader_source = generate_csr_spmv_shader(dtype)?;
    let module_name = format!("csr_spmv_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5, // row_ptrs, col_indices, values, x, y
        num_uniform_buffers: 1, // params
    });

    let pipeline = cache.get_or_create_dynamic_pipeline("csr_spmv", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[row_ptrs, col_indices, values, x, y, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_spmv"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_spmv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSR SpMM kernel: C = A * B
///
/// Element-parallel implementation where each thread computes one output element C[row, col].
///
/// # Buffers
///
/// - `row_ptrs`: CSR row pointers [m + 1] (I32)
/// - `col_indices`: CSR column indices [nnz] (I32)
/// - `a_values`: CSR values of A [nnz] (dtype)
/// - `b`: Dense matrix B [k, n] (dtype, row-major)
/// - `c`: Dense output matrix C [m, n] (dtype, row-major)
/// - `params`: Uniform buffer with SpmmParams { m, k, n }
pub fn launch_csr_spmm(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    a_values: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csr_spmm_{}", suffix);

    let shader_source = generate_csr_spmm_shader(dtype)?;
    let module_name = format!("csr_spmm_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5, // row_ptrs, col_indices, a_values, b, c
        num_uniform_buffers: 1, // params
    });

    let pipeline = cache.get_or_create_dynamic_pipeline("csr_spmm", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[row_ptrs, col_indices, a_values, b, c, params_buffer],
    );

    let total_elements = m * n;

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_spmm"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_spmm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSR extract diagonal kernel: diag[i] = A[i,i]
///
/// # Buffers
///
/// - `row_ptrs`: CSR row pointers [n + 1] (I32)
/// - `col_indices`: CSR column indices [nnz] (I32)
/// - `values`: CSR values [nnz] (dtype)
/// - `diag`: Output diagonal [n] (dtype)
/// - `params`: Uniform buffer with DiagParams { n }
pub fn launch_csr_extract_diagonal(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    values: &Buffer,
    diag: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csr_extract_diagonal_{}", suffix);

    let shader_source = generate_csr_extract_diagonal_shader(dtype)?;
    let module_name = format!("csr_extract_diagonal_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4, // row_ptrs, col_indices, values, diag
        num_uniform_buffers: 1, // params
    });

    let pipeline = cache.get_or_create_dynamic_pipeline(
        "csr_extract_diagonal",
        &entry_point,
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(
        &layout,
        &[row_ptrs, col_indices, values, diag, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_extract_diagonal"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_extract_diagonal"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_wgsl_syntax(source: &str) -> std::result::Result<(), String> {
        use wgpu::naga::front::wgsl;
        let mut frontend = wgsl::Frontend::new();
        frontend
            .parse(source)
            .map(|_| ())
            .map_err(|e| format!("WGSL parse error: {e}"))
    }

    #[test]
    fn test_csr_spmv_shader_syntax_f32() {
        let shader = generate_csr_spmv_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpMV shader should be valid WGSL");
    }

    #[test]
    fn test_csr_spmm_shader_syntax_f32() {
        let shader = generate_csr_spmm_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpMM shader should be valid WGSL");
    }
}
