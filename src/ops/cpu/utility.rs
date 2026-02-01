//! CPU implementation of utility operations.

use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::ops::UtilityOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
    kernels,
};
use crate::runtime::validate_arange;
use crate::tensor::Tensor;

/// UtilityOps implementation for CPU runtime.
impl UtilityOps<CpuRuntime> for CpuClient {
    fn clamp(
        &self,
        a: &Tensor<CpuRuntime>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let out_ptr = out.storage().ptr();
        let numel = a.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::clamp_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    numel,
                    min_val,
                    max_val,
                );
            }
        }, "clamp");

        Ok(out)
    }

    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<CpuRuntime>> {
        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device);
        let out_ptr = out.storage().ptr();
        let numel = out.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::fill_kernel::<T>(
                    out_ptr as *mut T,
                    T::from_f64(value),
                    numel,
                );
            }
        }, "fill");

        Ok(out)
    }

    fn arange(&self, start: f64, stop: f64, step: f64, dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Use shared validation
        let numel = validate_arange(start, stop, step)?;

        // Handle empty tensor case
        if numel == 0 {
            return Ok(Tensor::<CpuRuntime>::empty(&[0], dtype, &self.device));
        }

        let out = Tensor::<CpuRuntime>::empty(&[numel], dtype, &self.device);
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::arange_kernel::<T>(out_ptr as *mut T, start, step, numel);
            }
        }, "arange");

        Ok(out)
    }

    fn linspace(
        &self,
        start: f64,
        stop: f64,
        steps: usize,
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        // linspace supports all numeric dtypes - computation is done in f64,
        // then converted to the output dtype. This matches NumPy behavior.

        // Handle edge cases
        if steps == 0 {
            return Ok(Tensor::<CpuRuntime>::empty(&[0], dtype, &self.device));
        }

        if steps == 1 {
            let out = Tensor::<CpuRuntime>::empty(&[1], dtype, &self.device);
            let out_ptr = out.storage().ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    *(out_ptr as *mut T) = T::from_f64(start);
                }
            }, "linspace");

            return Ok(out);
        }

        let out = Tensor::<CpuRuntime>::empty(&[steps], dtype, &self.device);
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::linspace_kernel::<T>(out_ptr as *mut T, start, stop, steps);
            }
        }, "linspace");

        Ok(out)
    }

    fn eye(&self, n: usize, m: Option<usize>, dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Use shared validation
        use crate::runtime::validate_eye;
        let (rows, cols) = validate_eye(n, m);

        // Handle edge cases
        if rows == 0 || cols == 0 {
            return Ok(Tensor::<CpuRuntime>::empty(
                &[rows, cols],
                dtype,
                &self.device,
            ));
        }

        let out = Tensor::<CpuRuntime>::empty(&[rows, cols], dtype, &self.device);
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::eye_kernel::<T>(out_ptr as *mut T, rows, cols);
            }
        }, "eye");

        Ok(out)
    }
}
