//! WebGPU sparse triangular solve implementation.

use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferUsages};

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::generator::sparse_linalg::{
    generate_sparse_trsv_lower_shader, generate_sparse_trsv_upper_shader,
};
use super::super::{WgpuClient, WgpuRuntime};
use super::common::{WORKGROUP_SIZE, create_trsv_layout, validate_wgpu_dtype};
use crate::algorithm::sparse_linalg::validate_triangular_solve_dims;
use crate::algorithm::sparse_linalg_levels::{
    compute_levels_lower, compute_levels_upper, flatten_levels,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Sparse triangular solve for WebGPU.
pub fn sparse_solve_triangular_wgpu(
    client: &WgpuClient,
    l_or_u: &CsrData<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let (n, nrhs) = validate_triangular_solve_dims(l_or_u.shape, b.shape())?;
    let dtype = l_or_u.values().dtype();
    validate_wgpu_dtype(dtype, "sparse_solve_triangular")?;

    if b.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: b.dtype(),
        });
    }

    if nrhs > 1 {
        return Err(Error::Internal(
            "WebGPU sparse triangular solve only supports single RHS for now".to_string(),
        ));
    }

    // Extract CSR data for level analysis
    let row_ptrs: Vec<i64> = l_or_u.row_ptrs().to_vec();
    let col_indices: Vec<i64> = l_or_u.col_indices().to_vec();

    // Compute level schedule
    let schedule = if lower {
        compute_levels_lower(n, &row_ptrs, &col_indices)?
    } else {
        compute_levels_upper(n, &row_ptrs, &col_indices)?
    };
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

    // Allocate output and copy b into it on GPU (must be separate buffer)
    let x = Tensor::<WgpuRuntime>::zeros(b.shape(), dtype, &client.device_id);
    let copy_size = b.numel() * dtype.size_in_bytes();
    WgpuRuntime::copy_within_device(
        b.storage().ptr(),
        x.storage().ptr(),
        copy_size,
        &client.device_id,
    );

    // Process each level
    for level in 0..schedule.num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = level_end - level_start;

        if level_size == 0 {
            continue;
        }

        if lower {
            launch_sparse_trsv_lower(
                client,
                &level_rows_gpu,
                level_start,
                level_size,
                &row_ptrs_gpu,
                &col_indices_gpu,
                l_or_u.values(),
                b,
                &x,
                n,
                unit_diagonal,
            )?;
        } else {
            launch_sparse_trsv_upper(
                client,
                &level_rows_gpu,
                level_start,
                level_size,
                &row_ptrs_gpu,
                &col_indices_gpu,
                l_or_u.values(),
                b,
                &x,
                n,
            )?;
        }
    }

    client.poll_wait();

    Ok(x)
}

/// Launch sparse lower triangular solve kernel.
#[allow(clippy::too_many_arguments)]
fn launch_sparse_trsv_lower(
    client: &WgpuClient,
    level_rows: &Tensor<WgpuRuntime>,
    level_start: usize,
    level_size: usize,
    row_ptrs: &Tensor<WgpuRuntime>,
    col_indices: &Tensor<WgpuRuntime>,
    values: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    x: &Tensor<WgpuRuntime>,
    n: usize,
    unit_diagonal: bool,
) -> Result<()> {
    let shader_source = generate_sparse_trsv_lower_shader(DType::F32)?;
    let module = client
        .pipeline_cache
        .get_or_create_module_from_source("sparse_trsv_lower_f32", &shader_source);

    let layout = create_trsv_layout(&client.wgpu_device);

    let pipeline = client.pipeline_cache.get_or_create_dynamic_pipeline(
        "sparse_trsv_lower_f32",
        "sparse_trsv_lower_level_f32",
        &module,
        &layout,
    );

    let params: [u32; 4] = [
        level_size as u32,
        n as u32,
        if unit_diagonal { 1 } else { 0 },
        level_start as u32,
    ];
    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trsv_lower_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let level_rows_buf = get_tensor_buffer(level_rows)?;
    let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
    let col_indices_buf = get_tensor_buffer(col_indices)?;
    let values_buf = get_tensor_buffer(values)?;
    let b_buf = get_tensor_buffer(b)?;
    let x_buf = get_tensor_buffer(x)?;

    let bind_group = client.wgpu_device.create_bind_group(&BindGroupDescriptor {
        label: Some("trsv_lower_bind_group"),
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
                resource: b_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: x_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
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
            label: Some("sparse_trsv_lower"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    client.queue.submit(Some(encoder.finish()));

    Ok(())
}

/// Launch sparse upper triangular solve kernel.
#[allow(clippy::too_many_arguments)]
fn launch_sparse_trsv_upper(
    client: &WgpuClient,
    level_rows: &Tensor<WgpuRuntime>,
    level_start: usize,
    level_size: usize,
    row_ptrs: &Tensor<WgpuRuntime>,
    col_indices: &Tensor<WgpuRuntime>,
    values: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    x: &Tensor<WgpuRuntime>,
    n: usize,
) -> Result<()> {
    let shader_source = generate_sparse_trsv_upper_shader(DType::F32)?;
    let module = client
        .pipeline_cache
        .get_or_create_module_from_source("sparse_trsv_upper_f32", &shader_source);

    let layout = create_trsv_layout(&client.wgpu_device);

    let pipeline = client.pipeline_cache.get_or_create_dynamic_pipeline(
        "sparse_trsv_upper_f32",
        "sparse_trsv_upper_level_f32",
        &module,
        &layout,
    );

    let params: [u32; 4] = [level_size as u32, n as u32, 0, level_start as u32];
    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trsv_upper_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let level_rows_buf = get_tensor_buffer(level_rows)?;
    let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
    let col_indices_buf = get_tensor_buffer(col_indices)?;
    let values_buf = get_tensor_buffer(values)?;
    let b_buf = get_tensor_buffer(b)?;
    let x_buf = get_tensor_buffer(x)?;

    let bind_group = client.wgpu_device.create_bind_group(&BindGroupDescriptor {
        label: Some("trsv_upper_bind_group"),
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
                resource: b_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: x_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
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
            label: Some("sparse_trsv_upper"),
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
    fn test_sparse_solve_triangular_lower() {
        let client = get_client();
        let device = &client.device_id;

        // L = [2 0 0]
        //     [1 3 0]
        //     [0 2 4]
        let row_ptrs = Tensor::<WgpuRuntime>::from_slice(&[0i64, 1, 3, 5], &[4], device);
        let col_indices = Tensor::<WgpuRuntime>::from_slice(&[0i64, 0, 1, 1, 2], &[5], device);
        let values = Tensor::<WgpuRuntime>::from_slice(&[2.0f32, 1.0, 3.0, 2.0, 4.0], &[5], device);

        let l = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        let b = Tensor::<WgpuRuntime>::from_slice(&[2.0f32, 4.0, 8.0], &[3], device);

        let x = client
            .sparse_solve_triangular(&l, &b, true, false)
            .expect("Triangular solve should succeed");

        let x_data: Vec<f32> = x.to_vec();
        assert!((x_data[0] - 1.0).abs() < 1e-4);
        assert!((x_data[1] - 1.0).abs() < 1e-4);
        assert!((x_data[2] - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_sparse_solve_triangular_upper() {
        let client = get_client();
        let device = &client.device_id;

        // U = [2 1 0]
        //     [0 3 2]
        //     [0 0 4]
        let row_ptrs = Tensor::<WgpuRuntime>::from_slice(&[0i64, 2, 4, 5], &[4], device);
        let col_indices = Tensor::<WgpuRuntime>::from_slice(&[0i64, 1, 1, 2, 2], &[5], device);
        let values = Tensor::<WgpuRuntime>::from_slice(&[2.0f32, 1.0, 3.0, 2.0, 4.0], &[5], device);

        let u = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        let b = Tensor::<WgpuRuntime>::from_slice(&[5.0f32, 7.0, 8.0], &[3], device);

        let x = client
            .sparse_solve_triangular(&u, &b, false, false)
            .expect("Triangular solve should succeed");

        let x_data: Vec<f32> = x.to_vec();
        assert!((x_data[0] - 2.0).abs() < 1e-4);
        assert!((x_data[1] - 1.0).abs() < 1e-4);
        assert!((x_data[2] - 2.0).abs() < 1e-4);
    }
}
