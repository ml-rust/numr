//! Cumulative operations for CUDA runtime
use crate::error::{Error, Result};
use crate::ops::{CumulativeOps, reduce_dim_output_shape, reduce_output_shape};
use crate::runtime::cuda::kernels::{
    launch_cumprod, launch_cumprod_strided, launch_cumsum, launch_cumsum_strided, launch_logsumexp,
    launch_logsumexp_strided,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl CumulativeOps<CudaRuntime> for CudaClient {
    fn cumsum(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let shape = a.shape();
        let ndim = shape.len();

        // Normalize dimension (handle negative indexing)
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        // Handle empty tensor
        if a.numel() == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device));
        }

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(a);

        // Calculate dimensions for kernel launch
        let scan_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device);

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Scan along last dimension or effectively contiguous
            let outer = outer_size.max(1);
            unsafe {
                launch_cumsum(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    scan_size,
                    outer,
                )?;
            }
        } else {
            // Strided scan for non-last dimension
            unsafe {
                launch_cumsum_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    scan_size,
                    outer_size.max(1),
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    fn cumprod(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let shape = a.shape();
        let ndim = shape.len();

        // Normalize dimension (handle negative indexing)
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        // Handle empty tensor
        if a.numel() == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device));
        }

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(a);

        // Calculate dimensions for kernel launch
        let scan_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device);

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Scan along last dimension or effectively contiguous
            let outer = outer_size.max(1);
            unsafe {
                launch_cumprod(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    scan_size,
                    outer,
                )?;
            }
        } else {
            // Strided scan for non-last dimension
            unsafe {
                launch_cumprod_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    scan_size,
                    outer_size.max(1),
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    fn logsumexp(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        // Only support floating point types
        use crate::dtype::DType;
        if !matches!(a.dtype(), DType::F32 | DType::F64) {
            return Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "logsumexp",
            });
        }

        let shape = a.shape();
        let ndim = shape.len();

        // Handle empty dims (reduce over all dimensions)
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..ndim).collect()
        } else {
            dims.to_vec()
        };

        // Validate dimensions
        for &dim in &actual_dims {
            if dim >= ndim {
                return Err(Error::InvalidDimension {
                    dim: dim as isize,
                    ndim,
                });
            }
        }

        // Handle empty tensor
        if a.numel() == 0 {
            let out_shape = reduce_output_shape(shape, &actual_dims, keepdim);
            return Ok(Tensor::<CudaRuntime>::empty(
                &out_shape,
                a.dtype(),
                &self.device,
            ));
        }

        // For multi-dimensional reduction, reduce one dimension at a time
        if actual_dims.len() > 1 {
            let mut result = a.clone();
            // Sort dims in descending order to avoid index invalidation
            let mut sorted_dims = actual_dims.clone();
            sorted_dims.sort_by(|a, b| b.cmp(a));

            for &dim in &sorted_dims {
                result = self.logsumexp(&result, &[dim], true)?;
            }

            // Remove keepdim if not requested
            if !keepdim {
                let out_shape = reduce_output_shape(shape, &actual_dims, false);
                result = result.reshape(&out_shape)?;
            }

            return Ok(result);
        }

        // Single dimension reduction
        let dim = actual_dims[0];

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(a);

        // Calculate dimensions for kernel launch
        let reduce_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Calculate output shape
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);
        let out_numel: usize = out_shape.iter().product();

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(&out_shape, a.dtype(), &self.device);

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Reduction along last dimension
            let outer = outer_size.max(1);
            unsafe {
                launch_logsumexp(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    reduce_size,
                    outer,
                )?;
            }
        } else {
            // Strided reduction for non-last dimension
            unsafe {
                launch_logsumexp_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    reduce_size,
                    outer_size.max(1),
                    inner_size,
                )?;
            }
        }

        // Handle keepdim reshape if needed
        if keepdim && out.numel() == out_numel {
            Ok(out)
        } else {
            Ok(out)
        }
    }
}
