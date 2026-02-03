//! Gather and scatter operations for CUDA runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    launch_copy, launch_fill_with_f64, launch_gather, launch_index_put, launch_index_select,
    launch_scatter, launch_validate_indices,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{Runtime, compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;

/// Execute gather operation along dimension.
pub fn gather(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Validate dimension
    let ndim = a.ndim();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate index tensor has same number of dimensions
    if index.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: index.shape().to_vec(),
        });
    }

    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);

    // Output has same shape as index
    let out_shape = index.shape().to_vec();
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    // Prepare shape and stride arrays for GPU
    let input_shape: Vec<u32> = a.shape().iter().map(|&s| s as u32).collect();
    let input_strides: Vec<u32> = compute_contiguous_strides(a.shape())
        .iter()
        .map(|&s| s as u32)
        .collect();
    let output_shape: Vec<u32> = out_shape.iter().map(|&s| s as u32).collect();
    let output_strides: Vec<u32> = compute_contiguous_strides(&out_shape)
        .iter()
        .map(|&s| s as u32)
        .collect();

    // Allocate device memory for shape/stride arrays
    let shape_bytes = ndim * std::mem::size_of::<u32>();
    let input_shape_ptr = CudaRuntime::allocate(shape_bytes, &client.device);
    let input_strides_ptr = CudaRuntime::allocate(shape_bytes, &client.device);
    let output_shape_ptr = CudaRuntime::allocate(shape_bytes, &client.device);
    let output_strides_ptr = CudaRuntime::allocate(shape_bytes, &client.device);

    // Copy shape/stride data to device
    let input_shape_bytes: &[u8] = bytemuck::cast_slice(&input_shape);
    let input_strides_bytes: &[u8] = bytemuck::cast_slice(&input_strides);
    let output_shape_bytes: &[u8] = bytemuck::cast_slice(&output_shape);
    let output_strides_bytes: &[u8] = bytemuck::cast_slice(&output_strides);

    CudaRuntime::copy_to_device(input_shape_bytes, input_shape_ptr, &client.device);
    CudaRuntime::copy_to_device(input_strides_bytes, input_strides_ptr, &client.device);
    CudaRuntime::copy_to_device(output_shape_bytes, output_shape_ptr, &client.device);
    CudaRuntime::copy_to_device(output_strides_bytes, output_strides_ptr, &client.device);

    let result = unsafe {
        launch_gather(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.storage().ptr(),
            index_contig.storage().ptr(),
            out.storage().ptr(),
            ndim,
            dim,
            input_shape_ptr,
            input_strides_ptr,
            output_shape_ptr,
            output_strides_ptr,
            out.numel(),
        )
    };

    // Clean up temporary device allocations
    CudaRuntime::deallocate(input_shape_ptr, shape_bytes, &client.device);
    CudaRuntime::deallocate(input_strides_ptr, shape_bytes, &client.device);
    CudaRuntime::deallocate(output_shape_ptr, shape_bytes, &client.device);
    CudaRuntime::deallocate(output_strides_ptr, shape_bytes, &client.device);

    result?;
    Ok(out)
}

/// Execute scatter operation along dimension.
pub fn scatter(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Validate dimension
    let ndim = a.ndim();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate src has same dtype as input
    let dtype = a.dtype();
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Index and src must have same shape
    if index.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: index.shape().to_vec(),
            got: src.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);
    let src_contig = ensure_contiguous(src);

    // Output has same shape as input
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device);

    // First, copy input to output (scatter modifies output in-place)
    unsafe {
        launch_copy(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.storage().ptr(),
            out.storage().ptr(),
            a.numel(),
        )?;
    }

    // Prepare shape and stride arrays for GPU
    let output_shape: Vec<u32> = a.shape().iter().map(|&s| s as u32).collect();
    let output_strides: Vec<u32> = compute_contiguous_strides(a.shape())
        .iter()
        .map(|&s| s as u32)
        .collect();
    let src_shape: Vec<u32> = src.shape().iter().map(|&s| s as u32).collect();
    let src_strides: Vec<u32> = compute_contiguous_strides(src.shape())
        .iter()
        .map(|&s| s as u32)
        .collect();

    // Allocate device memory for shape/stride arrays
    let shape_bytes = ndim * std::mem::size_of::<u32>();
    let output_shape_ptr = CudaRuntime::allocate(shape_bytes, &client.device);
    let output_strides_ptr = CudaRuntime::allocate(shape_bytes, &client.device);
    let src_shape_ptr = CudaRuntime::allocate(shape_bytes, &client.device);
    let src_strides_ptr = CudaRuntime::allocate(shape_bytes, &client.device);

    // Copy shape/stride data to device
    CudaRuntime::copy_to_device(
        bytemuck::cast_slice(&output_shape),
        output_shape_ptr,
        &client.device,
    );
    CudaRuntime::copy_to_device(
        bytemuck::cast_slice(&output_strides),
        output_strides_ptr,
        &client.device,
    );
    CudaRuntime::copy_to_device(
        bytemuck::cast_slice(&src_shape),
        src_shape_ptr,
        &client.device,
    );
    CudaRuntime::copy_to_device(
        bytemuck::cast_slice(&src_strides),
        src_strides_ptr,
        &client.device,
    );

    let result = unsafe {
        launch_scatter(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.storage().ptr(),
            index_contig.storage().ptr(),
            src_contig.storage().ptr(),
            out.storage().ptr(),
            ndim,
            dim,
            output_shape_ptr,
            output_strides_ptr,
            src_shape_ptr,
            src_strides_ptr,
            src.numel(),
        )
    };

    // Clean up temporary device allocations
    CudaRuntime::deallocate(output_shape_ptr, shape_bytes, &client.device);
    CudaRuntime::deallocate(output_strides_ptr, shape_bytes, &client.device);
    CudaRuntime::deallocate(src_shape_ptr, shape_bytes, &client.device);
    CudaRuntime::deallocate(src_strides_ptr, shape_bytes, &client.device);

    result?;
    Ok(out)
}

/// Execute index_select operation.
pub fn index_select(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Validate index is 1D
    if index.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index.numel()],
            got: index.shape().to_vec(),
        });
    }

    // Validate dimension
    let shape = a.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);

    // Compute output shape: same as input but dim[dim] = index.len()
    let index_len = index.numel();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = index_len;

    // Compute dim_size for validation
    let dim_size = shape[dim];

    // Validate indices on GPU (only costs copying 4 bytes back)
    let error_count_tensor = Tensor::<CudaRuntime>::empty(&[1], DType::U32, &client.device);
    unsafe {
        // Initialize error count to 0
        launch_fill_with_f64(
            &client.context,
            &client.stream,
            client.device.index,
            DType::U32,
            0.0,
            error_count_tensor.storage().ptr(),
            1,
        )?;

        // Run validation kernel
        launch_validate_indices(
            &client.context,
            &client.stream,
            client.device.index,
            index_contig.storage().ptr(),
            error_count_tensor.storage().ptr(),
            index_len,
            dim_size,
        )?;
    }

    // Check validation result
    let error_count = error_count_tensor.to_vec::<u32>()[0];
    if error_count > 0 {
        return Err(Error::IndexOutOfBounds {
            index: 0, // We don't know which specific index failed
            size: dim_size,
        });
    }

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    // Compute outer/dim/inner sizes
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let outer_size = outer_size.max(1);
    let inner_size = inner_size.max(1);

    unsafe {
        launch_index_select(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.storage().ptr(),
            index_contig.storage().ptr(),
            out.storage().ptr(),
            outer_size,
            dim_size,
            inner_size,
            index_len,
        )?;
    }

    Ok(out)
}

/// Execute index_put operation.
pub fn index_put(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Validate index is 1D
    if index.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index.numel()],
            got: index.shape().to_vec(),
        });
    }

    // Validate src dtype matches
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    let index_len = index.numel();

    // Validate src shape: must match a's shape except at dim where it equals index_len
    let mut expected_src_shape = shape.to_vec();
    expected_src_shape[dim] = index_len;
    if src.shape() != expected_src_shape {
        return Err(Error::ShapeMismatch {
            expected: expected_src_shape,
            got: src.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);
    let src_contig = ensure_contiguous(src);

    // Compute dim_size for validation
    let dim_size = shape[dim];

    // Validate indices on GPU (only costs copying 4 bytes back)
    let error_count_tensor = Tensor::<CudaRuntime>::empty(&[1], DType::U32, &client.device);
    unsafe {
        // Initialize error count to 0
        launch_fill_with_f64(
            &client.context,
            &client.stream,
            client.device.index,
            DType::U32,
            0.0,
            error_count_tensor.storage().ptr(),
            1,
        )?;

        // Run validation kernel
        launch_validate_indices(
            &client.context,
            &client.stream,
            client.device.index,
            index_contig.storage().ptr(),
            error_count_tensor.storage().ptr(),
            index_len,
            dim_size,
        )?;
    }

    // Check validation result
    let error_count = error_count_tensor.to_vec::<u32>()[0];
    if error_count > 0 {
        return Err(Error::IndexOutOfBounds {
            index: 0, // We don't know which specific index failed
            size: dim_size,
        });
    }

    // Clone a to output first
    let out = a_contig.clone();

    // Compute outer/dim/inner sizes
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let outer_size = outer_size.max(1);
    let inner_size = inner_size.max(1);

    unsafe {
        launch_index_put(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            index_contig.storage().ptr(),
            src_contig.storage().ptr(),
            out.storage().ptr(),
            outer_size,
            dim_size,
            inner_size,
            index_len,
        )?;
    }

    Ok(out)
}
