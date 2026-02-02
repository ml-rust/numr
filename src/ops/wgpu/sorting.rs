//! Sorting operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::SortingOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    CountParams, FlatToMultiParams, SearchsortedParams, SortParams, TopkParams, alloc_output,
    create_params_buffer, get_tensor_buffer,
};
use crate::runtime::wgpu::shaders::sort;
use crate::runtime::{RuntimeClient, ensure_contiguous, normalize_dim};
use crate::tensor::Tensor;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, MapMode, PollType};

impl SortingOps<WgpuRuntime> for WgpuClient {
    fn sort(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // Check dtype support (WebGPU: F32, I32, U32)
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType { dtype, op: "sort" });
        }

        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Ok(a.clone());
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let sort_size = shape[dim_idx];

        // Check sort size limit (WebGPU bitonic sort in shared memory)
        if sort_size > crate::runtime::wgpu::shaders::generator::MAX_SHARED_SORT_SIZE {
            return Err(Error::backend_limitation(
                "WebGPU",
                "sort",
                format!(
                    "max {} elements per dimension, got {}",
                    crate::runtime::wgpu::shaders::generator::MAX_SHARED_SORT_SIZE,
                    sort_size
                ),
            ));
        }

        // Compute strides
        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        // Ensure contiguous
        let a_contig = ensure_contiguous(a);

        // Allocate output
        let out = alloc_output(self, shape, dtype);
        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params buffer
        let params = SortParams {
            outer_size: outer_size as u32,
            sort_size: sort_size as u32,
            inner_size: inner_size as u32,
            descending: descending as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        // Create dummy indices buffer
        let dummy_indices_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_sort_indices"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        sort::launch_sort_values_only(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            outer_size,
            inner_size,
            dtype,
        )?;

        drop(dummy_indices_buf);
        Ok(out)
    }

    fn sort_with_indices(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sort_with_indices",
            });
        }

        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            let indices = Tensor::zeros(&[], DType::I32, self.device());
            return Ok((a.clone(), indices));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let sort_size = shape[dim_idx];

        if sort_size > crate::runtime::wgpu::shaders::generator::MAX_SHARED_SORT_SIZE {
            return Err(Error::backend_limitation(
                "WebGPU",
                "sort_with_indices",
                format!(
                    "max {} elements per dimension, got {}",
                    crate::runtime::wgpu::shaders::generator::MAX_SHARED_SORT_SIZE,
                    sort_size
                ),
            ));
        }

        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        let a_contig = ensure_contiguous(a);

        let values_out = alloc_output(self, shape, dtype);
        let indices_out = alloc_output(self, shape, DType::I32);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let values_buf = get_tensor_buffer(&values_out)?;
        let indices_buf = get_tensor_buffer(&indices_out)?;

        let params = SortParams {
            outer_size: outer_size as u32,
            sort_size: sort_size as u32,
            inner_size: inner_size as u32,
            descending: descending as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        sort::launch_sort(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &values_buf,
            &indices_buf,
            &params_buf,
            outer_size,
            inner_size,
            dtype,
        )?;

        Ok((values_out, indices_out))
    }

    fn argsort(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "argsort",
            });
        }

        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Ok(Tensor::zeros(&[], DType::I32, self.device()));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let sort_size = shape[dim_idx];

        if sort_size > crate::runtime::wgpu::shaders::generator::MAX_SHARED_SORT_SIZE {
            return Err(Error::backend_limitation(
                "WebGPU",
                "argsort",
                format!(
                    "max {} elements per dimension, got {}",
                    crate::runtime::wgpu::shaders::generator::MAX_SHARED_SORT_SIZE,
                    sort_size
                ),
            ));
        }

        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        let a_contig = ensure_contiguous(a);

        let indices_out = alloc_output(self, shape, DType::I32);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let indices_buf = get_tensor_buffer(&indices_out)?;

        let params = SortParams {
            outer_size: outer_size as u32,
            sort_size: sort_size as u32,
            inner_size: inner_size as u32,
            descending: descending as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        sort::launch_argsort(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &indices_buf,
            &params_buf,
            outer_size,
            inner_size,
            dtype,
        )?;

        Ok(indices_out)
    }

    fn topk(
        &self,
        a: &Tensor<WgpuRuntime>,
        k: usize,
        dim: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType { dtype, op: "topk" });
        }

        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Err(Error::InvalidArgument {
                arg: "tensor",
                reason: "topk requires at least 1-D tensor".to_string(),
            });
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let sort_size = shape[dim_idx];

        if k == 0 || k > sort_size {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!("k must be in [1, {}], got {}", sort_size, k),
            });
        }

        if sort_size > crate::runtime::wgpu::shaders::generator::MAX_SHARED_SORT_SIZE {
            return Err(Error::backend_limitation(
                "WebGPU",
                "topk",
                format!(
                    "max {} elements per dimension, got {}",
                    crate::runtime::wgpu::shaders::generator::MAX_SHARED_SORT_SIZE,
                    sort_size
                ),
            ));
        }

        let outer_size: usize = shape[..dim_idx].iter().product();
        let inner_size: usize = shape[dim_idx + 1..].iter().product();
        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        let a_contig = ensure_contiguous(a);

        // Output shape has k instead of sort_size on dim
        let mut out_shape = shape.to_vec();
        out_shape[dim_idx] = k;

        let values_out = alloc_output(self, &out_shape, dtype);
        let indices_out = alloc_output(self, &out_shape, DType::I32);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let values_buf = get_tensor_buffer(&values_out)?;
        let indices_buf = get_tensor_buffer(&indices_out)?;

        let params = TopkParams {
            outer_size: outer_size as u32,
            sort_size: sort_size as u32,
            inner_size: inner_size as u32,
            k: k as u32,
            largest: largest as u32,
            sorted: sorted as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        sort::launch_topk(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &values_buf,
            &indices_buf,
            &params_buf,
            outer_size,
            inner_size,
            dtype,
        )?;

        Ok((values_out, indices_out))
    }

    fn unique(&self, a: &Tensor<WgpuRuntime>, _sorted: bool) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "unique",
            });
        }

        let numel = a.numel();

        if numel == 0 {
            return Ok(Tensor::empty(&[0], dtype, self.device()));
        }

        // Step 1: Flatten and sort
        let flat = a.reshape(&[numel])?;
        let sorted_tensor = self.sort(&flat, 0, false)?;

        // Step 2: Count unique elements
        let count_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("unique_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero initialize
        self.wgpu_queue()
            .write_buffer(&count_buf, 0, bytemuck::cast_slice(&[0u32]));

        let sorted_buf = get_tensor_buffer(&sorted_tensor)?;
        let params = CountParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        sort::launch_count_unique(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &sorted_buf,
            &count_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        // Read count
        let count = read_u32_from_buffer(self, &count_buf)?;

        if count == 0 {
            return Ok(Tensor::empty(&[0], dtype, self.device()));
        }

        // Step 3: Extract unique elements
        let out = alloc_output(self, &[count as usize], dtype);
        let out_buf = get_tensor_buffer(&out)?;

        // Reset counter
        self.wgpu_queue()
            .write_buffer(&count_buf, 0, bytemuck::cast_slice(&[0u32]));

        sort::launch_extract_unique(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &sorted_buf,
            &out_buf,
            &count_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn unique_with_counts(
        &self,
        _a: &Tensor<WgpuRuntime>,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        // WebGPU unique_with_counts requires complex stream compaction not easily done in WGSL
        // Return unsupported for now - can be implemented with multiple passes
        Err(Error::NotImplemented {
            feature: "unique_with_counts for WebGPU",
        })
    }

    fn nonzero(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "nonzero",
            });
        }

        let shape = a.shape();
        let ndim = shape.len();
        let numel = a.numel();

        if numel == 0 {
            return Ok(Tensor::empty(&[0, ndim], DType::I32, self.device()));
        }

        let a_contig = ensure_contiguous(a);
        let a_buf = get_tensor_buffer(&a_contig)?;

        // Phase 1: Count nonzero elements
        let count_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("nonzero_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.wgpu_queue()
            .write_buffer(&count_buf, 0, bytemuck::cast_slice(&[0u32]));

        let params = CountParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        sort::launch_count_nonzero(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &count_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        let nnz = read_u32_from_buffer(self, &count_buf)? as usize;

        if nnz == 0 {
            return Ok(Tensor::empty(&[0, ndim], DType::I32, self.device()));
        }

        // Phase 2: Gather flat indices
        let flat_indices = alloc_output(self, &[nnz], DType::I32);
        let flat_indices_buf = get_tensor_buffer(&flat_indices)?;

        // Reset counter
        self.wgpu_queue()
            .write_buffer(&count_buf, 0, bytemuck::cast_slice(&[0u32]));

        sort::launch_gather_nonzero(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &a_buf,
            &flat_indices_buf,
            &count_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        // Phase 3: Convert flat indices to multi-indices
        let multi_indices = alloc_output(self, &[nnz, ndim], DType::I32);
        let multi_indices_buf = get_tensor_buffer(&multi_indices)?;

        // Create shape buffer
        let mut shape_arr = [0u32; 8];
        for (i, &s) in shape.iter().enumerate().take(8) {
            shape_arr[i] = s as u32;
        }

        let flat_to_multi_params = FlatToMultiParams {
            nnz: nnz as u32,
            ndim: ndim as u32,
            _pad0: 0,
            _pad1: 0,
            shape: shape_arr,
        };
        let flat_to_multi_params_buf = create_params_buffer(self, &flat_to_multi_params);

        sort::launch_flat_to_multi_index(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &flat_indices_buf,
            &multi_indices_buf,
            &flat_to_multi_params_buf,
            nnz,
        )?;

        Ok(multi_indices)
    }

    fn searchsorted(
        &self,
        sorted_sequence: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        right: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = sorted_sequence.dtype();

        if dtype != values.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: values.dtype(),
            });
        }

        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "searchsorted",
            });
        }

        // Sequence must be 1-D
        if sorted_sequence.shape().len() != 1 {
            return Err(Error::InvalidArgument {
                arg: "sorted_sequence",
                reason: "sorted_sequence must be 1-D".to_string(),
            });
        }

        let seq_len = sorted_sequence.numel();
        let num_values = values.numel();

        if num_values == 0 {
            return Ok(Tensor::empty(values.shape(), DType::I32, self.device()));
        }

        let seq_contig = ensure_contiguous(sorted_sequence);
        let values_contig = ensure_contiguous(values);

        let out = alloc_output(self, values.shape(), DType::I32);

        let seq_buf = get_tensor_buffer(&seq_contig)?;
        let values_buf = get_tensor_buffer(&values_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = SearchsortedParams {
            seq_len: seq_len as u32,
            num_values: num_values as u32,
            right: right as u32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        sort::launch_searchsorted(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &seq_buf,
            &values_buf,
            &out_buf,
            &params_buf,
            num_values,
            dtype,
        )?;

        Ok(out)
    }
}

// Helper function to read u32 from GPU buffer
fn read_u32_from_buffer(client: &WgpuClient, buffer: &Buffer) -> Result<u32> {
    let mut encoder = client
        .wgpu_device()
        .create_command_encoder(&Default::default());

    let staging_buffer = client.wgpu_device().create_buffer(&BufferDescriptor {
        label: Some("staging_buffer"),
        size: 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, 4);
    let submission = client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    slice.map_async(MapMode::Read, move |result| {
        tx.send(result).ok();
    });

    let _ = client.wgpu_device().poll(PollType::Wait {
        timeout: Some(std::time::Duration::from_secs(30)),
        submission_index: Some(submission),
    });
    rx.recv()
        .ok()
        .and_then(|r| r.ok())
        .ok_or_else(|| crate::error::Error::Backend("Failed to read buffer".to_string()))?;

    let data = slice.get_mapped_range();
    let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    drop(data);
    staging_buffer.unmap();

    Ok(value)
}
