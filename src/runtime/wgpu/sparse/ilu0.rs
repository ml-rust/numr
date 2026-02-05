//! WebGPU ILU(0) factorization implementation.

use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferUsages};

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::generator::sparse_linalg::{
    generate_find_diag_indices_shader, generate_ilu0_level_shader,
};
use super::super::{WgpuClient, WgpuRuntime};
use super::common::{WORKGROUP_SIZE, create_ilu_ic_layout, split_lu_wgpu, validate_wgpu_dtype};
use crate::algorithm::sparse_linalg::{
    IluDecomposition, IluOptions, SymbolicIlu0, validate_square_sparse,
};
use crate::algorithm::sparse_linalg::{compute_levels_ilu, flatten_levels};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// ILU(0) factorization for WebGPU.
pub fn ilu0_wgpu(
    client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
    options: IluOptions,
) -> Result<IluDecomposition<WgpuRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_wgpu_dtype(dtype, "ilu0")?;

    // Extract CSR data for level analysis (unavoidable for dependency analysis)
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();

    // Compute level schedule
    let schedule = compute_levels_ilu(n, &row_ptrs, &col_indices)?;
    let (level_ptrs, level_rows) = flatten_levels(&schedule);

    // Convert to i32 for GPU
    let level_rows_i32: Vec<i32> = level_rows.iter().map(|&x| x as i32).collect();
    let row_ptrs_i32: Vec<i32> = row_ptrs.iter().map(|&x| x as i32).collect();
    let col_indices_i32: Vec<i32> = col_indices.iter().map(|&x| x as i32).collect();

    // Create GPU buffers
    let level_rows_gpu = Tensor::<WgpuRuntime>::from_slice(
        &level_rows_i32,
        &[level_rows_i32.len()],
        &client.device_id,
    );
    let row_ptrs_gpu =
        Tensor::<WgpuRuntime>::from_slice(&row_ptrs_i32, &[row_ptrs_i32.len()], &client.device_id);
    let col_indices_gpu = Tensor::<WgpuRuntime>::from_slice(
        &col_indices_i32,
        &[col_indices_i32.len()],
        &client.device_id,
    );

    // Clone values for in-place factorization
    let values_gpu = a.values().clone();

    // Allocate diagonal indices buffer
    let diag_indices_gpu = Tensor::<WgpuRuntime>::zeros(&[n], DType::I32, &client.device_id);

    // Find diagonal indices on GPU
    launch_find_diag_indices(
        client,
        &row_ptrs_gpu,
        &col_indices_gpu,
        &diag_indices_gpu,
        n,
    )?;

    // Process each level
    for level in 0..schedule.num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = level_end - level_start;

        if level_size == 0 {
            continue;
        }

        launch_ilu0_level(
            client,
            &level_rows_gpu,
            level_start,
            level_size,
            &row_ptrs_gpu,
            &col_indices_gpu,
            &values_gpu,
            &diag_indices_gpu,
            n,
            options.diagonal_shift as f32,
        )?;
    }

    // Wait for GPU to complete
    client.poll_wait();

    // Split into L and U
    split_lu_wgpu(client, n, &row_ptrs, &col_indices, &values_gpu)
}

/// ILU(0) symbolic factorization for WebGPU (runs on CPU, returns reusable symbolic data).
///
/// The symbolic phase analyzes the sparsity pattern and precomputes the update schedule.
/// This can be reused for multiple numeric factorizations with different values.
pub fn ilu0_symbolic_wgpu(
    _client: &WgpuClient,
    pattern: &CsrData<WgpuRuntime>,
) -> Result<SymbolicIlu0> {
    let n = validate_square_sparse(pattern.shape)?;

    // Extract CSR structure for CPU-based symbolic analysis
    // This transfer is acceptable as symbolic analysis happens once per matrix structure
    let row_ptrs: Vec<i64> = pattern.row_ptrs().to_vec();
    let col_indices: Vec<i64> = pattern.col_indices().to_vec();

    // Delegate to shared implementation (pure CPU graph analysis)
    crate::algorithm::sparse_linalg::ilu0_symbolic_impl(n, &row_ptrs, &col_indices)
}

/// ILU(0) numeric factorization for WebGPU using precomputed symbolic data.
///
/// Uses the level schedule derived from the symbolic data for parallel execution.
pub fn ilu0_numeric_wgpu(
    client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
    symbolic: &SymbolicIlu0,
    options: IluOptions,
) -> Result<IluDecomposition<WgpuRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_wgpu_dtype(dtype, "ilu0")?;

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

    // Extract CSR data - must match symbolic pattern
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();

    // Compute level schedule from the pattern
    let schedule = compute_levels_ilu(n, &row_ptrs, &col_indices)?;
    let (level_ptrs, level_rows) = flatten_levels(&schedule);

    // Convert to i32 for GPU
    let level_rows_i32: Vec<i32> = level_rows.iter().map(|&x| x as i32).collect();
    let row_ptrs_i32: Vec<i32> = row_ptrs.iter().map(|&x| x as i32).collect();
    let col_indices_i32: Vec<i32> = col_indices.iter().map(|&x| x as i32).collect();

    // Create GPU buffers
    let level_rows_gpu = Tensor::<WgpuRuntime>::from_slice(
        &level_rows_i32,
        &[level_rows_i32.len()],
        &client.device_id,
    );
    let row_ptrs_gpu =
        Tensor::<WgpuRuntime>::from_slice(&row_ptrs_i32, &[row_ptrs_i32.len()], &client.device_id);
    let col_indices_gpu = Tensor::<WgpuRuntime>::from_slice(
        &col_indices_i32,
        &[col_indices_i32.len()],
        &client.device_id,
    );

    // Clone values for in-place factorization
    let values_gpu = a.values().clone();

    // Allocate diagonal indices buffer
    let diag_indices_gpu = Tensor::<WgpuRuntime>::zeros(&[n], DType::I32, &client.device_id);

    // Find diagonal indices on GPU
    launch_find_diag_indices(
        client,
        &row_ptrs_gpu,
        &col_indices_gpu,
        &diag_indices_gpu,
        n,
    )?;

    // Process each level
    for level in 0..schedule.num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = level_end - level_start;

        if level_size == 0 {
            continue;
        }

        launch_ilu0_level(
            client,
            &level_rows_gpu,
            level_start,
            level_size,
            &row_ptrs_gpu,
            &col_indices_gpu,
            &values_gpu,
            &diag_indices_gpu,
            n,
            options.diagonal_shift as f32,
        )?;
    }

    // Wait for GPU to complete
    client.poll_wait();

    // Split into L and U
    split_lu_wgpu(client, n, &row_ptrs, &col_indices, &values_gpu)
}

/// Launch find diagonal indices kernel.
pub(super) fn launch_find_diag_indices(
    client: &WgpuClient,
    row_ptrs: &Tensor<WgpuRuntime>,
    col_indices: &Tensor<WgpuRuntime>,
    diag_indices: &Tensor<WgpuRuntime>,
    n: usize,
) -> Result<()> {
    let shader_source = generate_find_diag_indices_shader();
    let module = client
        .pipeline_cache
        .get_or_create_module_from_source("find_diag_indices", &shader_source);

    // Create bind group layout
    let layout = client
        .wgpu_device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("find_diag_indices_layout"),
            entries: &[
                // row_ptrs (read-only input)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // col_indices (read-only input)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // diag_indices (output)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline = client.pipeline_cache.get_or_create_dynamic_pipeline(
        "find_diag_indices",
        "find_diag_indices",
        &module,
        &layout,
    );

    // Create params buffer
    let params = [n as u32, 0u32, 0u32, 0u32];
    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("find_diag_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    // Get buffers
    let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
    let col_indices_buf = get_tensor_buffer(col_indices)?;
    let diag_indices_buf = get_tensor_buffer(diag_indices)?;

    // Create bind group
    let bind_group = client.wgpu_device.create_bind_group(&BindGroupDescriptor {
        label: Some("find_diag_indices_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: row_ptrs_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: col_indices_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: diag_indices_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let workgroups = (n as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("find_diag_indices"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    client.queue.submit(Some(encoder.finish()));

    Ok(())
}

/// Launch ILU0 level kernel.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_ilu0_level(
    client: &WgpuClient,
    level_rows: &Tensor<WgpuRuntime>,
    level_start: usize,
    level_size: usize,
    row_ptrs: &Tensor<WgpuRuntime>,
    col_indices: &Tensor<WgpuRuntime>,
    values: &Tensor<WgpuRuntime>,
    diag_indices: &Tensor<WgpuRuntime>,
    n: usize,
    diagonal_shift: f32,
) -> Result<()> {
    let shader_source = generate_ilu0_level_shader(DType::F32)?;
    let module = client
        .pipeline_cache
        .get_or_create_module_from_source("ilu0_level_f32", &shader_source);

    let layout = create_ilu_ic_layout(&client.wgpu_device);

    let pipeline = client.pipeline_cache.get_or_create_dynamic_pipeline(
        "ilu0_level_f32",
        "ilu0_level_f32",
        &module,
        &layout,
    );

    // Create params buffer: level_size, n, diagonal_shift, level_start
    let params: [u32; 4] = [
        level_size as u32,
        n as u32,
        diagonal_shift.to_bits(),
        level_start as u32,
    ];
    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ilu0_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    // Get buffers
    let level_rows_buf = get_tensor_buffer(level_rows)?;
    let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
    let col_indices_buf = get_tensor_buffer(col_indices)?;
    let values_buf = get_tensor_buffer(values)?;
    let diag_indices_buf = get_tensor_buffer(diag_indices)?;

    let bind_group = client.wgpu_device.create_bind_group(&BindGroupDescriptor {
        label: Some("ilu0_level_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: level_rows_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: row_ptrs_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: col_indices_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: values_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: diag_indices_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let workgroups = (level_size as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ilu0_level"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    client.queue.submit(Some(encoder.finish()));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms;
    use crate::runtime::Runtime;

    fn get_client() -> WgpuClient {
        let device = WgpuRuntime::default_device();
        WgpuRuntime::default_client(&device)
    }

    #[test]
    fn test_ilu0_basic() {
        let client = get_client();
        let device = &client.device_id;

        // Create a simple 3x3 sparse matrix in CSR format
        let row_ptrs = Tensor::<WgpuRuntime>::from_slice(&[0i64, 2, 5, 7], &[4], device);
        let col_indices =
            Tensor::<WgpuRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2], &[7], device);
        let values = Tensor::<WgpuRuntime>::from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[7],
            device,
        );

        let a = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        let decomp = client
            .ilu0(&a, IluOptions::default())
            .expect("ILU0 should succeed");

        assert_eq!(decomp.l.shape, [3, 3]);
        assert_eq!(decomp.u.shape, [3, 3]);
    }
}
