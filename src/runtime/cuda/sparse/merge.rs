//! Sparse matrix element-wise operations (add, sub, mul, div)
//!
//! GPU-native implementations for CSR, CSC, and COO formats using two-pass merge algorithms.

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::cuda::kernels::{
    coo_add_merge, coo_div_merge, coo_mul_merge, coo_sub_merge, csc_add_merge, csc_div_merge,
    csc_mul_merge, csc_sub_merge, csr_add_merge, csr_div_merge, csr_mul_merge, csr_sub_merge,
};
use crate::tensor::Tensor;

impl CudaClient {
    // =========================================================================
    // CSR Operations
    // =========================================================================

    pub(crate) fn add_csr_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [nrows, _] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            csr_add_merge,
            dtype,
            "CSR addition",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_row_ptrs,
                a_col_indices,
                a_values,
                b_row_ptrs,
                b_col_indices,
                b_values,
                nrows,
            )
        )
    }

    pub(crate) fn sub_csr_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [nrows, _] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            csr_sub_merge,
            dtype,
            "CSR subtraction",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_row_ptrs,
                a_col_indices,
                a_values,
                b_row_ptrs,
                b_col_indices,
                b_values,
                nrows,
            )
        )
    }

    pub(crate) fn mul_csr_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [nrows, _] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            csr_mul_merge,
            dtype,
            "CSR multiplication",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_row_ptrs,
                a_col_indices,
                a_values,
                b_row_ptrs,
                b_col_indices,
                b_values,
                nrows,
            )
        )
    }

    pub(crate) fn div_csr_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [nrows, _] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            csr_div_merge,
            dtype,
            "CSR division",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_row_ptrs,
                a_col_indices,
                a_values,
                b_row_ptrs,
                b_col_indices,
                b_values,
                nrows,
            )
        )
    }

    // =========================================================================
    // CSC Operations
    // =========================================================================

    pub(crate) fn add_csc_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [_, ncols] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            csc_add_merge,
            dtype,
            "CSC addition",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_col_ptrs,
                a_row_indices,
                a_values,
                b_col_ptrs,
                b_row_indices,
                b_values,
                ncols,
            )
        )
    }

    pub(crate) fn sub_csc_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [_, ncols] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            csc_sub_merge,
            dtype,
            "CSC subtraction",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_col_ptrs,
                a_row_indices,
                a_values,
                b_col_ptrs,
                b_row_indices,
                b_values,
                ncols,
            )
        )
    }

    pub(crate) fn mul_csc_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [_, ncols] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            csc_mul_merge,
            dtype,
            "CSC multiplication",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_col_ptrs,
                a_row_indices,
                a_values,
                b_col_ptrs,
                b_row_indices,
                b_values,
                ncols,
            )
        )
    }

    pub(crate) fn div_csc_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [_, ncols] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            csc_div_merge,
            dtype,
            "CSC division",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_col_ptrs,
                a_row_indices,
                a_values,
                b_col_ptrs,
                b_row_indices,
                b_values,
                ncols,
            )
        )
    }

    // =========================================================================
    // COO Operations
    // =========================================================================

    pub(crate) fn add_coo_impl<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            coo_add_merge,
            dtype,
            "COO addition",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_row_indices,
                a_col_indices,
                a_values,
                b_row_indices,
                b_col_indices,
                b_values,
                shape,
            )
        )
    }

    pub(crate) fn sub_coo_impl<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            coo_sub_merge,
            dtype,
            "COO subtraction",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_row_indices,
                a_col_indices,
                a_values,
                b_row_indices,
                b_col_indices,
                b_values,
                shape,
            )
        )
    }

    pub(crate) fn mul_coo_impl<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            coo_mul_merge,
            dtype,
            "COO multiplication",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_row_indices,
                a_col_indices,
                a_values,
                b_row_indices,
                b_col_indices,
                b_values,
                shape,
            )
        )
    }

    pub(crate) fn div_coo_impl<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let device = a_values.device();
        let dtype = a_values.dtype();

        sparse_dtype_dispatch!(
            coo_div_merge,
            dtype,
            "COO division",
            (
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                a_row_indices,
                a_col_indices,
                a_values,
                b_row_indices,
                b_col_indices,
                b_values,
                shape,
            )
        )
    }
}
