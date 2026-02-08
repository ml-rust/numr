//! SVD decomposition for CUDA
//!
//! All operations run entirely on GPU with zero CPU transfers.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{SvdDecomposition, validate_linalg_dtype, validate_matrix_2d};
use crate::error::Result;
use crate::runtime::{AllocGuard, Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// SVD decomposition via Jacobi method - runs entirely on GPU.
pub fn svd_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<SvdDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let k = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    // Handle empty matrix
    if m == 0 || n == 0 {
        let u_ptr = client.allocator().allocate(0)?;
        let s_ptr = client.allocator().allocate(0)?;
        let vt_ptr = client.allocator().allocate(0)?;
        let u = unsafe { CudaClient::tensor_from_raw(u_ptr, &[m, k], dtype, device) };
        let s = unsafe { CudaClient::tensor_from_raw(s_ptr, &[k], dtype, device) };
        let vt = unsafe { CudaClient::tensor_from_raw(vt_ptr, &[k, n], dtype, device) };
        return Ok(SvdDecomposition { u, s, vt });
    }

    // If m < n, transpose and swap U/V at the end
    let transpose = m < n;
    let (work_m, work_n) = if transpose { (n, m) } else { (m, n) };
    let work_k = work_m.min(work_n);

    // Allocate working buffers on GPU
    let elem_size = dtype.size_in_bytes();
    let b_size = work_m * work_n * elem_size;
    let v_size = work_n * work_n * elem_size;
    let s_size = work_n * elem_size;
    let flag_size = std::mem::size_of::<i32>();

    let b_guard = AllocGuard::new(client.allocator(), b_size)?;
    let v_guard = AllocGuard::new(client.allocator(), v_size)?;
    let s_guard = AllocGuard::new(client.allocator(), s_size)?;
    let converged_flag_guard = AllocGuard::new(client.allocator(), flag_size)?;

    let b_ptr = b_guard.ptr();
    let v_ptr = v_guard.ptr();
    let s_ptr = s_guard.ptr();
    let converged_flag_ptr = converged_flag_guard.ptr();

    // Copy input to B, transposing if needed using GPU transpose kernel
    if transpose {
        // Use optimized GPU transpose: A[m,n] -> B[n,m]
        let result = unsafe {
            kernels::launch_transpose(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                a.storage().ptr(),
                b_ptr,
                m, // rows of input
                n, // cols of input
            )
        };
        result?
    } else {
        CudaRuntime::copy_within_device(a.storage().ptr(), b_ptr, b_size, device)?;
    }

    // Zero-initialize converged flag
    let zero_i32: [u8; 4] = [0; 4];
    CudaRuntime::copy_to_device(&zero_i32, converged_flag_ptr, device)?;

    // Launch SVD Jacobi kernel
    let result = unsafe {
        kernels::launch_svd_jacobi(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            b_ptr,
            v_ptr,
            s_ptr,
            converged_flag_ptr,
            work_m,
            work_n,
        )
    };

    result?;

    client.synchronize();

    // GPU argsort to get sorted indices (descending order)
    let indices_size = work_n * std::mem::size_of::<i64>();
    let indices_guard = AllocGuard::new(client.allocator(), indices_size)?;
    let indices_ptr = indices_guard.ptr();

    let argsort_result = unsafe {
        kernels::launch_argsort(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            s_ptr,       // input: singular values
            indices_ptr, // output: sorted indices
            1,           // outer_size
            work_n,      // sort_size
            1,           // inner_size
            true,        // descending (largest singular values first)
        )
    };

    argsort_result?;

    // Now reorder S, U, V using GPU index_select
    // S_sorted: select first work_k elements using indices
    let s_sorted_size = work_k * elem_size;
    let s_sorted_guard = AllocGuard::new(client.allocator(), s_sorted_size)?;
    let s_sorted_ptr = s_sorted_guard.ptr();

    let s_select_result = unsafe {
        kernels::launch_index_select(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            s_ptr,        // input
            indices_ptr,  // indices
            s_sorted_ptr, // output
            1,            // outer_size
            work_n,       // dim_size
            1,            // inner_size
            work_k,       // index_len (first k indices)
        )
    };

    s_select_result?;

    // U_sorted: B is [work_m, work_n], select work_k columns -> [work_m, work_k]
    let u_sorted_size = work_m * work_k * elem_size;
    let u_sorted_guard = AllocGuard::new(client.allocator(), u_sorted_size)?;
    let u_sorted_ptr = u_sorted_guard.ptr();

    let u_select_result = unsafe {
        kernels::launch_index_select(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            b_ptr,        // input [work_m, work_n]
            indices_ptr,  // indices
            u_sorted_ptr, // output [work_m, work_k]
            work_m,       // outer_size (rows)
            work_n,       // dim_size (columns to select from)
            1,            // inner_size
            work_k,       // index_len (first k indices)
        )
    };

    u_select_result?;

    // V_sorted: V is [work_n, work_n], select work_k columns -> [work_n, work_k]
    let v_sorted_size = work_n * work_k * elem_size;
    let v_sorted_guard = AllocGuard::new(client.allocator(), v_sorted_size)?;
    let v_sorted_ptr = v_sorted_guard.ptr();

    let v_select_result = unsafe {
        kernels::launch_index_select(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            v_ptr,        // input [work_n, work_n]
            indices_ptr,  // indices
            v_sorted_ptr, // output [work_n, work_k]
            work_n,       // outer_size (rows)
            work_n,       // dim_size (columns to select from)
            1,            // inner_size
            work_k,       // index_len (first k indices)
        )
    };

    v_select_result?;

    // VT = transpose(V_sorted): [work_n, work_k] -> [work_k, work_n]
    let vt_size = work_k * work_n * elem_size;
    let vt_guard = AllocGuard::new(client.allocator(), vt_size)?;
    let vt_ptr = vt_guard.ptr();

    let vt_transpose_result = unsafe {
        kernels::launch_transpose(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            v_sorted_ptr, // input [work_n, work_k]
            vt_ptr,       // output [work_k, work_n]
            work_n,       // rows of input
            work_k,       // cols of input
        )
    };

    vt_transpose_result?;

    // Create final tensors based on transpose flag
    let (u_final, s_final, vt_final) = if transpose {
        // When we transposed input: swap roles of U and VT
        // Original: A^T = U @ S @ VT
        // So: A = VT^T @ S @ U^T = V @ S @ U^T
        // Therefore: U_out = V (from VT^T), VT_out = U^T

        // VT^T = V: [work_k, work_n] -> need to transpose to get [m, k] = [work_n, work_k]
        // But our vt_ptr is [work_k, work_n], we need its transpose [work_n, work_k]
        // which is [n, k] = [m, k] since m < n means work_n = m

        // Actually, when transpose=true: work_m = n, work_n = m
        // So VT is [work_k, work_n] = [k, m]
        // We need U_out [m, k] which is transpose of VT: [k, m]^T = [m, k]
        let u_out_size = m * k * elem_size;
        let u_out_guard = AllocGuard::new(client.allocator(), u_out_size)?;
        let u_out_ptr = u_out_guard.ptr();

        let u_out_transpose_result = unsafe {
            kernels::launch_transpose(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                vt_ptr,    // input [k, m] (since work_k=k, work_n=m)
                u_out_ptr, // output [m, k]
                work_k,    // rows of input = k
                work_n,    // cols of input = m
            )
        };

        u_out_transpose_result?;

        // VT_out = U^T: U_sorted is [work_m, work_k] = [n, k]
        // We need VT_out [k, n] which is transpose of U_sorted
        let vt_out_size = k * n * elem_size;
        let vt_out_guard = AllocGuard::new(client.allocator(), vt_out_size)?;
        let vt_out_ptr = vt_out_guard.ptr();

        let vt_out_transpose_result = unsafe {
            kernels::launch_transpose(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                u_sorted_ptr, // input [n, k] (since work_m=n, work_k=k)
                vt_out_ptr,   // output [k, n]
                work_m,       // rows of input = n
                work_k,       // cols of input = k
            )
        };

        vt_out_transpose_result?;

        let u =
            unsafe { CudaClient::tensor_from_raw(u_out_guard.release(), &[m, k], dtype, device) };
        let s =
            unsafe { CudaClient::tensor_from_raw(s_sorted_guard.release(), &[k], dtype, device) };
        let vt =
            unsafe { CudaClient::tensor_from_raw(vt_out_guard.release(), &[k, n], dtype, device) };

        (u, s, vt)
    } else {
        // No transpose case:
        // - u_sorted_ptr [m, k] is the final U
        // - s_sorted_ptr [k] is the final S
        // - vt_ptr [k, n] is the final VT (already transposed from v_sorted)

        let u = unsafe {
            CudaClient::tensor_from_raw(u_sorted_guard.release(), &[m, k], dtype, device)
        };
        let s =
            unsafe { CudaClient::tensor_from_raw(s_sorted_guard.release(), &[k], dtype, device) };
        let vt = unsafe { CudaClient::tensor_from_raw(vt_guard.release(), &[k, n], dtype, device) };

        (u, s, vt)
    };

    Ok(SvdDecomposition {
        u: u_final,
        s: s_final,
        vt: vt_final,
    })
}
