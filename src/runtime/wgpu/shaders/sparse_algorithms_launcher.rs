//! Launcher functions for sparse algorithm WGSL shaders.
//!
//! Provides launch functions for:
//! - Column-parallel DSMM (Dense × Sparse Matrix Multiplication)
//! - SpGEMM symbolic and numeric phases

use wgpu::{Buffer, Queue};

use super::generator::dtype_suffix;
use super::generator::sparse_algorithms::{
    generate_dsmm_csc_shader, generate_spgemm_accumulate_shader, generate_spgemm_scatter_shader,
    generate_spgemm_symbolic_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

/// Launch DSMM (Dense × Sparse) kernel: C = A * B
///
/// Dense A [M, K] × Sparse B CSC [K, N] → Dense C [M, N]
///
/// # Buffers
///
/// - `a`: Dense matrix A [M, K] (dtype, row-major)
/// - `col_ptrs`: CSC column pointers [N + 1] (I32)
/// - `row_indices`: CSC row indices [nnz] (I32)
/// - `b_values`: CSC values [nnz] (dtype)
/// - `c`: Dense output matrix C [M, N] (dtype, row-major)
/// - `params_buffer`: Uniform buffer with DsmmParams { m, k, n }
pub fn launch_dsmm_csc(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    col_ptrs: &Buffer,
    row_indices: &Buffer,
    b_values: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("dsmm_csc_{}", suffix);

    let shader_source = generate_dsmm_csc_shader(dtype)?;
    let module_name = format!("dsmm_csc_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,  // a, col_ptrs, row_indices, b_values, c
        num_uniform_buffers: 1,  // params
        num_readonly_storage: 4, // a, col_ptrs, row_indices, b_values
    });

    let pipeline = cache.get_or_create_dynamic_pipeline("dsmm_csc", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[a, col_ptrs, row_indices, b_values, c, params_buffer],
    );

    let total_elements = m * n;

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dsmm_csc"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dsmm_csc"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch SpGEMM symbolic phase: count NNZ per output row
///
/// # Buffers
///
/// - `a_row_ptrs`: CSR row pointers for A [M + 1] (I32)
/// - `a_col_indices`: CSR column indices for A [nnz_a] (I32)
/// - `b_row_ptrs`: CSR row pointers for B [K + 1] (I32)
/// - `b_col_indices`: CSR column indices for B [nnz_b] (I32)
/// - `row_nnz`: Output NNZ per row [M] (I32)
/// - `params_buffer`: Uniform buffer with SpgemmSymbolicParams { m, n }
/// - `bitmap`: Working bitmap buffer [M * words_per_row] (atomic<u32>)
pub fn launch_spgemm_symbolic(
    cache: &PipelineCache,
    queue: &Queue,
    a_row_ptrs: &Buffer,
    a_col_indices: &Buffer,
    b_row_ptrs: &Buffer,
    b_col_indices: &Buffer,
    row_nnz: &Buffer,
    params_buffer: &Buffer,
    bitmap: &Buffer,
    m: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("spgemm_symbolic_{}", suffix);

    let shader_source = generate_spgemm_symbolic_shader(dtype)?;
    let module_name = format!("spgemm_symbolic_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6, // a_row_ptrs, a_col_indices, b_row_ptrs, b_col_indices, row_nnz, bitmap
        num_uniform_buffers: 1, // params
        num_readonly_storage: 4, // a_row_ptrs, a_col_indices, b_row_ptrs, b_col_indices
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("spgemm_symbolic", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_row_ptrs,
            a_col_indices,
            b_row_ptrs,
            b_col_indices,
            row_nnz,
            bitmap,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("spgemm_symbolic"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spgemm_symbolic"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(m), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch SpGEMM accumulate phase.
///
/// # Buffers
///
/// - `a_row_ptrs`: CSR row pointers for A [M + 1] (I32)
/// - `a_col_indices`: CSR column indices for A [nnz_a] (I32)
/// - `a_values`: CSR values for A [nnz_a] (dtype)
/// - `b_row_ptrs`: CSR row pointers for B [K + 1] (I32)
/// - `b_col_indices`: CSR column indices for B [nnz_b] (I32)
/// - `b_values`: CSR values for B [nnz_b] (dtype)
/// - `params_buffer`: Uniform buffer with SpgemmParams { m, n }
/// - `accum`: Dense accumulator [M * N] (dtype)
/// - `flags`: Column flags [M * N] (u32)
pub fn launch_spgemm_accumulate(
    cache: &PipelineCache,
    queue: &Queue,
    a_row_ptrs: &Buffer,
    a_col_indices: &Buffer,
    a_values: &Buffer,
    b_row_ptrs: &Buffer,
    b_col_indices: &Buffer,
    b_values: &Buffer,
    params_buffer: &Buffer,
    accum: &Buffer,
    flags: &Buffer,
    m: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("spgemm_accumulate_{}", suffix);

    let shader_source = generate_spgemm_accumulate_shader(dtype)?;
    let module_name = format!("spgemm_accumulate_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 8, // a_row_ptrs, a_col_indices, a_values, b_row_ptrs, b_col_indices, b_values, accum, flags
        num_uniform_buffers: 1, // params
        num_readonly_storage: 6, // a_row_ptrs, a_col_indices, a_values, b_row_ptrs, b_col_indices, b_values
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("spgemm_accumulate", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            accum,
            flags,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("spgemm_accumulate"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spgemm_accumulate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(m), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch SpGEMM scatter phase: compact accumulators into CSR arrays.
pub fn launch_spgemm_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    c_row_ptrs: &Buffer,
    accum: &Buffer,
    flags: &Buffer,
    c_col_indices: &Buffer,
    c_values: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("spgemm_scatter_{}", suffix);

    let shader_source = generate_spgemm_scatter_shader(dtype)?;
    let module_name = format!("spgemm_scatter_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,  // c_row_ptrs, accum, flags, c_col_indices, c_values
        num_uniform_buffers: 1,  // params
        num_readonly_storage: 3, // c_row_ptrs, accum, flags
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("spgemm_scatter", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            c_row_ptrs,
            accum,
            flags,
            c_col_indices,
            c_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("spgemm_scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spgemm_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(m), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::generator::sparse_algorithms::{
        generate_dsmm_csc_shader, generate_spgemm_accumulate_shader,
        generate_spgemm_scatter_shader, generate_spgemm_symbolic_shader,
    };
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
    fn test_dsmm_csc_shader_syntax_f32() {
        let shader = generate_dsmm_csc_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("DSMM shader should be valid WGSL");
    }

    #[test]
    fn test_spgemm_symbolic_shader_syntax_f32() {
        let shader = generate_spgemm_symbolic_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpGEMM symbolic shader should be valid WGSL");
    }

    #[test]
    fn test_spgemm_accumulate_shader_syntax_f32() {
        let shader = generate_spgemm_accumulate_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpGEMM accumulate shader should be valid WGSL");
    }

    #[test]
    fn test_spgemm_scatter_shader_syntax_f32() {
        let shader = generate_spgemm_scatter_shader(DType::F32).unwrap();
        validate_wgsl_syntax(&shader).expect("SpGEMM scatter shader should be valid WGSL");
    }
}
