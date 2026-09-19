//! Native tiled GEMM entry points for the CUDA client.
//!
//! Handles the operand preparation the kernels cannot: transpose-view
//! detection so a `[K,N]` view reaches GEMV, the small-M F32 kernel or the
//! transposed-B tiled GEMM without materialising the copy, batch
//! broadcasting, and the integer output-dtype widening.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::matmul_output_shape;
use crate::runtime::cuda::kernels::{
    MAX_SMALL_M, int_matmul_output_dtype, launch_gemv_kernel_bt_mr, launch_matmul_batched_kernel,
    launch_matmul_batched_kernel_bt, launch_matmul_batched_smallm_bt_kernel, launch_matmul_kernel,
    launch_matmul_kernel_bt, launch_matmul_smallm_bt_kernel,
};
use crate::runtime::cuda::ops::matmul_broadcast::resolve_batched_operands;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Native matrix multiplication using tiled CUDA kernel.
///
/// Uses shared memory tiling for cache efficiency. This is the default
/// implementation that works without any vendor dependencies.
/// Detect if a 2D tensor is a simple transpose of a contiguous [N,K] matrix.
///
/// A tensor with shape [K, N] and strides [1, K] is a transpose view of
/// contiguous [N, K] data. We can pass the raw pointer directly to gemv_bt
/// instead of materializing the transpose (which copies the entire matrix).
fn is_simple_transpose_2d(tensor: &Tensor<CudaRuntime>) -> bool {
    let shape = tensor.shape();
    let strides = tensor.strides();
    if shape.len() != 2 {
        return false;
    }
    // shape=[K,N], strides=[1,K] means transpose of contiguous [N,K]
    strides[0] == 1 && strides[1] == shape[0] as isize
}

/// Whether this dtype still takes the small-M GEMV path against a transposed
/// weight.
///
/// Only the half dtypes do, and only because their tensor-core kernel has no
/// transposed-B loader yet: past the GEMV rows the op copies the weight to
/// `[K, N]` per call, which is too slow for a decode step. The GEMV reduces
/// each output along K lane-strided and then through a shuffle tree, so a
/// row of a 17-row product is NOT the same bits as its 1-row product for
/// these dtypes. Closing that needs a WMMA tile that stages `B` from `[N, K]`
/// (`col_major` fragments) with a 16-row block for the decode regime.
///
/// F32 takes the transposed-B tiled kernel at every M: one FMA per k in k
/// order per element in every tile, so its rows are batch-invariant. F64 and
/// the integer dtypes take the generic tiled kernel at every M for the same
/// reason. FP8 has no GEMV kernel at all, and I8 could not use one: its matmul
/// widens to I32 (see `int_matmul_output_dtype`) while every GEMV kernel
/// writes the element type.
#[inline]
fn gemv_only_half(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

pub(crate) fn matmul_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Half dtypes against a transposed [N,K] weight at small M: the gemv_bt
    // kernel, which reads the weight in place. See `gemv_only_half` for why
    // this switch survives for them alone.
    if m <= 16 && gemv_only_half(dtype) && is_simple_transpose_2d(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_gemv_kernel_bt_mr(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(), // raw [N,K] pointer — no copy!
                out.ptr(),
                1, // batch
                m,
                n,
                k,
                1, // a_batch
                1, // b_batch
            )?;
        }

        return Ok(out);
    }

    // F32 against a transposed weight at M <= MAX_SMALL_M: one thread per
    // output, reading the `[N, K]` buffer in place. Same FMA chain per element
    // as the tiled kernel below, so the same bits (see `MAX_SMALL_M`); it
    // only skips the 16-row tile's padding rows. The launcher declines a
    // weight wider than `MAX_SMALL_N` and a grid past `SMALLM_MAX_WAVES`
    // waves of this device's SMs, where the tiled kernel's B reuse wins.
    if dtype == DType::F32 && m <= MAX_SMALL_M && is_simple_transpose_2d(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;
        let launched = unsafe {
            launch_matmul_smallm_bt_kernel(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(),
                out.ptr(),
                m,
                n,
                k,
            )?
        };
        if launched {
            return Ok(out);
        }
    }

    // F32 against a transposed weight, at every M: the tiled F32 kernel reads
    // the `[N, K]` buffer in place, and its 16-row tile serves the decode
    // shapes. Making `b` contiguous here would copy the whole weight on every
    // call. F32 only: the other dtypes have no transposed-B tile loader.
    if dtype == DType::F32 && is_simple_transpose_2d(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;
        let launched = unsafe {
            launch_matmul_kernel_bt(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(),
                out.ptr(),
                m,
                n,
                k,
            )?
        };
        if launched {
            return Ok(out);
        }
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    // I8 is the one dtype whose matmul does not write its own dtype: it widens
    // to I32, matching CPU's quantized accumulation. Every other dtype maps to
    // itself here.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}

/// Detect if the last two dims of a 3D tensor are a simple transpose.
/// Shape [B, K, N] with strides [B_stride, 1, K] means each batch slice
/// is a transpose of contiguous [N, K].
fn is_batched_transpose_last2(tensor: &Tensor<CudaRuntime>) -> bool {
    let shape = tensor.shape();
    let strides = tensor.strides();
    if shape.len() != 3 {
        return false;
    }
    let k = shape[1];
    let n = shape[2];
    // strides: [n*k, 1, k] means transpose of contiguous [batch, N, K]
    strides[1] == 1 && strides[2] == k as isize && strides[0] == (n * k) as isize
}

/// Native batched matrix multiplication using tiled CUDA kernel.
pub(crate) fn matmul_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Pointers and batch counts must come from the same tensors, so both are taken
    // from one resolver rather than derived separately.
    let operands = resolve_batched_operands(a, b, &out_shape)?;
    let (a, b) = (&operands.a, &operands.b);
    let (a_batch, b_batch) = (operands.a_batch, operands.b_batch);

    // Half dtypes against a transposed weight at small M: gemv_bt, for the
    // reason `gemv_only_half` gives.
    if m <= 16 && gemv_only_half(dtype) && is_batched_transpose_last2(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        unsafe {
            launch_gemv_kernel_bt_mr(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(),
                out.ptr(),
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            )?;
        }

        return Ok(out);
    }

    // Same small-M one-thread-per-output path as `matmul_native`, over a
    // transposed `[batch, N, K]` operand.
    if dtype == DType::F32 && m <= MAX_SMALL_M && is_batched_transpose_last2(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;
        let launched = unsafe {
            launch_matmul_batched_smallm_bt_kernel(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(),
                out.ptr(),
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            )?
        };
        if launched {
            return Ok(out);
        }
    }

    // Same in-place read of a transposed `[batch, N, K]` operand as
    // `matmul_native`.
    if dtype == DType::F32 && is_batched_transpose_last2(b) {
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;
        let launched = unsafe {
            launch_matmul_batched_kernel_bt(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                a_contig.ptr(),
                b.ptr(),
                out.ptr(),
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            )?
        };
        if launched {
            return Ok(out);
        }
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    // I8 widens to I32 here too — see `matmul_native`.
    let out =
        Tensor::<CudaRuntime>::empty(&out_shape, int_matmul_output_dtype(dtype), &client.device)?;

    unsafe {
        launch_matmul_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            b_contig.ptr(),
            out.ptr(),
            batch,
            m,
            n,
            k,
            a_batch,
            b_batch,
        )?;
    }

    Ok(out)
}
