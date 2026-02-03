//! Common utilities for WebGPU sparse linear algebra.

use std::time::Duration;

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ShaderStages,
};

use super::super::{WgpuClient, WgpuRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Workgroup size for sparse linear algebra kernels.
pub const WORKGROUP_SIZE: u32 = 256;

/// Validate dtype for WebGPU sparse linear algebra (F32 only).
pub fn validate_wgpu_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Create bind group layout for ILU/IC kernels.
pub fn create_ilu_ic_layout(device: &wgpu::Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ilu_ic_layout"),
        entries: &[
            // level_rows
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // row_ptrs
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // col_indices
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // values (read_write for in-place factorization)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // diag_indices
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // params (uniform)
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create bind group layout for triangular solve kernels.
pub fn create_trsv_layout(device: &wgpu::Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("trsv_layout"),
        entries: &[
            // level_rows
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // row_ptrs
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // col_indices
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // values
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // b
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // x (read_write)
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // params (uniform)
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

impl WgpuClient {
    /// Wait for GPU operations to complete.
    pub(super) fn poll_wait(&self) {
        let _ = self.wgpu_device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        });
    }
}

/// Split factored LU matrix into L and U components.
pub fn split_lu_wgpu(
    client: &WgpuClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<WgpuRuntime>,
) -> Result<crate::algorithm::sparse_linalg::IluDecomposition<WgpuRuntime>> {
    // Extract values to CPU for splitting (TODO: could be GPU kernel)
    let values: Vec<f32> = values_gpu.to_vec();

    // Split into L and U
    let mut l_row_ptrs = vec![0i64; n + 1];
    let mut l_col_indices = Vec::new();
    let mut l_values = Vec::new();
    let mut u_row_ptrs = vec![0i64; n + 1];
    let mut u_col_indices = Vec::new();
    let mut u_values = Vec::new();

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            let val = values[idx];

            if j < i {
                l_col_indices.push(j as i64);
                l_values.push(val);
            } else {
                u_col_indices.push(j as i64);
                u_values.push(val);
            }
        }

        l_row_ptrs[i + 1] = l_col_indices.len() as i64;
        u_row_ptrs[i + 1] = u_col_indices.len() as i64;
    }

    // Create output tensors
    let l_row_ptrs_t = Tensor::<WgpuRuntime>::from_slice(&l_row_ptrs, &[n + 1], &client.device_id);
    let l_col_indices_t = Tensor::<WgpuRuntime>::from_slice(
        &l_col_indices,
        &[l_col_indices.len()],
        &client.device_id,
    );
    let u_row_ptrs_t = Tensor::<WgpuRuntime>::from_slice(&u_row_ptrs, &[n + 1], &client.device_id);
    let u_col_indices_t = Tensor::<WgpuRuntime>::from_slice(
        &u_col_indices,
        &[u_col_indices.len()],
        &client.device_id,
    );

    let l_values_t =
        Tensor::<WgpuRuntime>::from_slice(&l_values, &[l_values.len()], &client.device_id);
    let u_values_t =
        Tensor::<WgpuRuntime>::from_slice(&u_values, &[u_values.len()], &client.device_id);

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_t, [n, n])?;
    let u = CsrData::new(u_row_ptrs_t, u_col_indices_t, u_values_t, [n, n])?;

    Ok(crate::algorithm::sparse_linalg::IluDecomposition { l, u })
}

/// Extract lower triangular matrix after IC factorization.
pub fn extract_lower_wgpu(
    client: &WgpuClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<WgpuRuntime>,
) -> Result<crate::algorithm::sparse_linalg::IcDecomposition<WgpuRuntime>> {
    // Extract values to CPU (TODO: could be GPU kernel)
    let values: Vec<f32> = values_gpu.to_vec();

    // Filter to lower triangle
    let mut new_row_ptrs = vec![0i64; n + 1];
    let mut new_col_indices = Vec::new();
    let mut new_values = Vec::new();

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j <= i {
                new_col_indices.push(j as i64);
                new_values.push(values[idx]);
            }
        }

        new_row_ptrs[i + 1] = new_col_indices.len() as i64;
    }

    // Create output tensors
    let l_row_ptrs_t =
        Tensor::<WgpuRuntime>::from_slice(&new_row_ptrs, &[n + 1], &client.device_id);
    let l_col_indices_t = Tensor::<WgpuRuntime>::from_slice(
        &new_col_indices,
        &[new_col_indices.len()],
        &client.device_id,
    );
    let l_values_t =
        Tensor::<WgpuRuntime>::from_slice(&new_values, &[new_values.len()], &client.device_id);

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_t, [n, n])?;

    Ok(crate::algorithm::sparse_linalg::IcDecomposition { l })
}
