//! Sparse matrix element-wise operations (add, sub, mul, div)
//!
//! GPU-native implementations for CSR, CSC, and COO formats using two-pass merge algorithms.

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::Element;
use crate::error::Result;
use crate::sparse::SparseOps;
use crate::tensor::Tensor;

impl SparseOps<CudaRuntime> for CudaClient {
    // =========================================================================
    // CSR Operations
    // =========================================================================

    fn add_csr<T: Element>(
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
        sparse_dtype_dispatch_csr!(
            csr_add_merge,
            self,
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            "CSR addition"
        )
    }

    fn sub_csr<T: Element>(
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
        sparse_dtype_dispatch_csr!(
            csr_sub_merge,
            self,
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            "CSR subtraction"
        )
    }

    fn mul_csr<T: Element>(
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
        sparse_dtype_dispatch_csr!(
            csr_mul_merge,
            self,
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            "CSR multiplication"
        )
    }

    fn div_csr<T: Element>(
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
        sparse_dtype_dispatch_csr!(
            csr_div_merge,
            self,
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            "CSR division"
        )
    }

    // =========================================================================
    // CSC Operations
    // =========================================================================

    fn add_csc<T: Element>(
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
        sparse_dtype_dispatch_csc!(
            csc_add_merge,
            self,
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            "CSC addition"
        )
    }

    fn sub_csc<T: Element>(
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
        sparse_dtype_dispatch_csc!(
            csc_sub_merge,
            self,
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            "CSC subtraction"
        )
    }

    fn mul_csc<T: Element>(
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
        sparse_dtype_dispatch_csc!(
            csc_mul_merge,
            self,
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            "CSC multiplication"
        )
    }

    fn div_csc<T: Element>(
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
        sparse_dtype_dispatch_csc!(
            csc_div_merge,
            self,
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            "CSC division"
        )
    }

    // =========================================================================
    // COO Operations
    // =========================================================================

    fn add_coo<T: Element>(
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
        sparse_dtype_dispatch_coo!(
            coo_add_merge,
            self,
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
            "COO addition"
        )
    }

    fn sub_coo<T: Element>(
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
        sparse_dtype_dispatch_coo!(
            coo_sub_merge,
            self,
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
            "COO subtraction"
        )
    }

    fn mul_coo<T: Element>(
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
        sparse_dtype_dispatch_coo!(
            coo_mul_merge,
            self,
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
            "COO multiplication"
        )
    }

    fn div_coo<T: Element>(
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
        sparse_dtype_dispatch_coo!(
            coo_div_merge,
            self,
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
            "COO division"
        )
    }
}
