//! Tensor construction and scalar-extraction helpers shared by statistics ops.

use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::common::statistics_common::compute_bin_edges_f64;
use crate::runtime::cpu::helpers::dispatch_dtype;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// Create bin edges tensor from computed f64 edges.
pub(crate) fn create_bin_edges(
    client: &CpuClient,
    min_val: f64,
    max_val: f64,
    bins: usize,
    dtype: DType,
) -> Result<Tensor<CpuRuntime>> {
    let edges_data = compute_bin_edges_f64(min_val, max_val, bins);

    // Create tensor and copy data based on dtype
    let edges = Tensor::<CpuRuntime>::empty(&[bins + 1], dtype, &client.device)?;
    let edges_ptr = edges.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            let out_slice = std::slice::from_raw_parts_mut(edges_ptr as *mut T, bins + 1);
            for (i, &val) in edges_data.iter().enumerate() {
                out_slice[i] = T::from_f64(val);
            }
        }
    }, "histogram_edges");

    Ok(edges)
}

/// Extract scalar f64 value from tensor.
pub(crate) fn tensor_to_f64(t: &Tensor<CpuRuntime>) -> Result<f64> {
    let dtype = t.dtype();
    let ptr = t.ptr();

    let val = dispatch_dtype!(dtype, T => {
        unsafe { (*(ptr as *const T)).to_f64() }
    }, "tensor_to_f64");

    Ok(val)
}
