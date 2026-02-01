//! Indexing operations for CUDA runtime
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{IndexingOps, compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::cuda::kernels::{
    launch_argmax_dim, launch_argmin_dim, launch_copy, launch_embedding_lookup,
    launch_fill_with_f64, launch_gather, launch_index_put, launch_index_select,
    launch_masked_count, launch_masked_fill, launch_masked_prefix_sum, launch_masked_select,
    launch_scatter, launch_validate_indices,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{Runtime, compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;

impl IndexingOps<CudaRuntime> for CudaClient {
    fn argmax(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        keepdim: bool,
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

        let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);

        unsafe {
            launch_argmax_dim(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                reduce_size,
                inner_size,
            )?;
        }

        Ok(out)
    }

    fn argmin(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        keepdim: bool,
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

        let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);

        unsafe {
            launch_argmin_dim(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                reduce_size,
                inner_size,
            )?;
        }

        Ok(out)
    }

    fn gather(
        &self,
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
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

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
        let input_shape_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let input_strides_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let output_shape_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let output_strides_ptr = CudaRuntime::allocate(shape_bytes, &self.device);

        // Copy shape/stride data to device
        let input_shape_bytes: &[u8] = bytemuck::cast_slice(&input_shape);
        let input_strides_bytes: &[u8] = bytemuck::cast_slice(&input_strides);
        let output_shape_bytes: &[u8] = bytemuck::cast_slice(&output_shape);
        let output_strides_bytes: &[u8] = bytemuck::cast_slice(&output_strides);

        CudaRuntime::copy_to_device(input_shape_bytes, input_shape_ptr, &self.device);
        CudaRuntime::copy_to_device(input_strides_bytes, input_strides_ptr, &self.device);
        CudaRuntime::copy_to_device(output_shape_bytes, output_shape_ptr, &self.device);
        CudaRuntime::copy_to_device(output_strides_bytes, output_strides_ptr, &self.device);

        let result = unsafe {
            launch_gather(
                &self.context,
                &self.stream,
                self.device.index,
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
        CudaRuntime::deallocate(input_shape_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(input_strides_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(output_shape_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(output_strides_ptr, shape_bytes, &self.device);

        result?;
        Ok(out)
    }

    fn scatter(
        &self,
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
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        // First, copy input to output (scatter modifies output in-place)
        unsafe {
            launch_copy(
                &self.context,
                &self.stream,
                self.device.index,
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
        let output_shape_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let output_strides_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let src_shape_ptr = CudaRuntime::allocate(shape_bytes, &self.device);
        let src_strides_ptr = CudaRuntime::allocate(shape_bytes, &self.device);

        // Copy shape/stride data to device
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&output_shape),
            output_shape_ptr,
            &self.device,
        );
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&output_strides),
            output_strides_ptr,
            &self.device,
        );
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&src_shape),
            src_shape_ptr,
            &self.device,
        );
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&src_strides),
            src_strides_ptr,
            &self.device,
        );

        let result = unsafe {
            launch_scatter(
                &self.context,
                &self.stream,
                self.device.index,
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
        CudaRuntime::deallocate(output_shape_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(output_strides_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(src_shape_ptr, shape_bytes, &self.device);
        CudaRuntime::deallocate(src_strides_ptr, shape_bytes, &self.device);

        result?;
        Ok(out)
    }

    fn index_select(
        &self,
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
        let error_count_tensor = Tensor::<CudaRuntime>::empty(&[1], DType::U32, &self.device);
        unsafe {
            // Initialize error count to 0
            launch_fill_with_f64(
                &self.context,
                &self.stream,
                self.device.index,
                DType::U32,
                0.0,
                error_count_tensor.storage().ptr(),
                1,
            )?;

            // Run validation kernel
            launch_validate_indices(
                &self.context,
                &self.stream,
                self.device.index,
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

        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        // Compute outer/dim/inner sizes
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        unsafe {
            launch_index_select(
                &self.context,
                &self.stream,
                self.device.index,
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

    fn index_put(
        &self,
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
        let error_count_tensor = Tensor::<CudaRuntime>::empty(&[1], DType::U32, &self.device);
        unsafe {
            // Initialize error count to 0
            launch_fill_with_f64(
                &self.context,
                &self.stream,
                self.device.index,
                DType::U32,
                0.0,
                error_count_tensor.storage().ptr(),
                1,
            )?;

            // Run validation kernel
            launch_validate_indices(
                &self.context,
                &self.stream,
                self.device.index,
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
                &self.context,
                &self.stream,
                self.device.index,
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

    fn masked_select(
        &self,
        a: &Tensor<CudaRuntime>,
        mask: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate mask dtype
        if mask.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: mask.dtype(),
            });
        }

        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let mask_contig = ensure_contiguous(mask);
        let numel = a.numel();

        // Both tensors must have same shape (or mask must broadcast to a's shape)
        // For simplicity, require same shape for now
        if a.shape() != mask.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: mask.shape().to_vec(),
            });
        }

        // Phase 1: Count true elements in mask
        let count_bytes = std::mem::size_of::<u32>();
        let count_ptr = CudaRuntime::allocate(count_bytes, &self.device);

        // Initialize count to 0
        let zero: u32 = 0;
        CudaRuntime::copy_to_device(bytemuck::bytes_of(&zero), count_ptr, &self.device);

        unsafe {
            launch_masked_count(
                &self.context,
                &self.stream,
                self.device.index,
                mask_contig.storage().ptr(),
                count_ptr,
                numel,
            )?;
        }

        // Read count back to host
        let mut count_buf = [0u32; 1];
        CudaRuntime::copy_from_device(
            count_ptr,
            bytemuck::bytes_of_mut(&mut count_buf),
            &self.device,
        );
        let count = count_buf[0] as usize;

        CudaRuntime::deallocate(count_ptr, count_bytes, &self.device);

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(&[count], dtype, &self.device);

        if count == 0 {
            return Ok(out);
        }

        // Phase 2: Compute prefix sum
        let prefix_sum_bytes = numel * std::mem::size_of::<u32>();
        let prefix_sum_ptr = CudaRuntime::allocate(prefix_sum_bytes, &self.device);

        unsafe {
            launch_masked_prefix_sum(
                &self.context,
                &self.stream,
                self.device.index,
                mask_contig.storage().ptr(),
                prefix_sum_ptr,
                numel,
            )?;
        }

        // Phase 3: Gather selected elements
        unsafe {
            launch_masked_select(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                mask_contig.storage().ptr(),
                out.storage().ptr(),
                prefix_sum_ptr,
                numel,
            )?;
        }

        CudaRuntime::deallocate(prefix_sum_ptr, prefix_sum_bytes, &self.device);

        Ok(out)
    }

    fn masked_fill(
        &self,
        a: &Tensor<CudaRuntime>,
        mask: &Tensor<CudaRuntime>,
        value: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate mask dtype
        if mask.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: mask.dtype(),
            });
        }

        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let mask_contig = ensure_contiguous(mask);

        // Both tensors must have same shape (or mask must broadcast to a's shape)
        // For simplicity, require same shape for now
        if a.shape() != mask.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: mask.shape().to_vec(),
            });
        }

        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_masked_fill(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                mask_contig.storage().ptr(),
                out.storage().ptr(),
                value,
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn embedding_lookup(
        &self,
        embeddings: &Tensor<CudaRuntime>,
        indices: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = embeddings.dtype();
        let emb_shape = embeddings.shape();

        // Validate embeddings is 2D
        if emb_shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                expected: vec![0, 0], // Indicates 2D expected
                got: emb_shape.to_vec(),
            });
        }

        // Validate indices dtype
        if indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: indices.dtype(),
            });
        }

        let vocab_size = emb_shape[0];
        let embedding_dim = emb_shape[1];
        let num_indices = indices.numel();

        // Output shape: indices.shape() + [embedding_dim]
        let mut out_shape = indices.shape().to_vec();
        out_shape.push(embedding_dim);

        let emb_contig = ensure_contiguous(embeddings);
        let idx_contig = ensure_contiguous(indices);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        unsafe {
            launch_embedding_lookup(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                emb_contig.storage().ptr(),
                idx_contig.storage().ptr(),
                out.storage().ptr(),
                num_indices,
                vocab_size,
                embedding_dim,
            )?;
        }

        Ok(out)
    }
}
