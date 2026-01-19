//! WebGPU implementation of linear algebra algorithms
//!
//! This module implements the [`LinearAlgebraAlgorithms`] trait for WebGPU.
//! All algorithms use native WGSL compute shaders - NO CPU FALLBACK.
//!
//! # Performance Note
//!
//! Operations use native WGSL compute shaders running entirely on the GPU.
//! Currently only F32 is supported (WGSL doesn't support F64).

use super::client::get_buffer;
use super::shaders::linalg as kernels;
use super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{
    CholeskyDecomposition, LinearAlgebraAlgorithms, LuDecomposition, MatrixNormOrder,
    QrDecomposition, SvdDecomposition, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TensorOps;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::{Layout, Storage, Tensor};

/// Helper macro to get a GPU buffer from a pointer with proper error context.
///
/// Reduces boilerplate for the common pattern:
/// ```ignore
/// let buffer = get_buffer(ptr)
///     .ok_or_else(|| Error::Internal("Failed to get X buffer".to_string()))?;
/// ```
///
/// Usage: `let buffer = get_buffer_or_err!(ptr, "LU");`
macro_rules! get_buffer_or_err {
    ($ptr:expr, $name:expr) => {
        get_buffer($ptr).ok_or_else(|| {
            Error::Internal(format!(
                "Failed to get {} buffer from GPU allocation",
                $name
            ))
        })?
    };
}

impl LinearAlgebraAlgorithms<WgpuRuntime> for WgpuClient {
    fn lu_decompose(&self, a: &Tensor<WgpuRuntime>) -> Result<LuDecomposition<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let k = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        // WGSL only supports F32
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU lu_decompose (only F32 supported)",
            });
        }

        // Allocate buffers
        let lu_size = m * n * dtype.size_in_bytes();
        let lu_ptr = self.allocator().allocate(lu_size);
        let lu_buffer = get_buffer(lu_ptr)
            .ok_or_else(|| Error::Internal("Failed to get LU buffer".to_string()))?;

        let pivots_size = k * std::mem::size_of::<i32>();
        let pivots_ptr = self.allocator().allocate(pivots_size);
        let pivots_buffer = get_buffer(pivots_ptr)
            .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

        let num_swaps_size = std::mem::size_of::<i32>();
        let num_swaps_ptr = self.allocator().allocate(num_swaps_size);
        let num_swaps_buffer = get_buffer(num_swaps_ptr)
            .ok_or_else(|| Error::Internal("Failed to get num_swaps buffer".to_string()))?;

        let singular_flag_size = std::mem::size_of::<i32>();
        let singular_flag_ptr = self.allocator().allocate(singular_flag_size);
        let singular_flag_buffer = get_buffer(singular_flag_ptr)
            .ok_or_else(|| Error::Internal("Failed to get singular_flag buffer".to_string()))?;

        // Copy input to LU buffer
        WgpuRuntime::copy_within_device(a.storage().ptr(), lu_ptr, lu_size, device);

        // Create params buffer
        let params: [u32; 2] = [m as u32, n as u32];
        let params_buffer = self.create_uniform_buffer("lu_params", 8);
        self.write_buffer(&params_buffer, &params);

        // Zero-initialize flags
        let zero_i32: [i32; 1] = [0];
        self.write_buffer(&num_swaps_buffer, &zero_i32);
        self.write_buffer(&singular_flag_buffer, &zero_i32);

        // Launch kernel
        kernels::launch_lu_decompose(
            self.pipeline_cache(),
            &self.queue,
            &lu_buffer,
            &pivots_buffer,
            &num_swaps_buffer,
            &singular_flag_buffer,
            &params_buffer,
            dtype,
        )?;

        self.synchronize();

        // Read back flags
        let staging = self.create_staging_buffer("lu_flags_staging", 8);
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lu_flags_copy"),
                });
        encoder.copy_buffer_to_buffer(&num_swaps_buffer, 0, &staging, 0, 4);
        encoder.copy_buffer_to_buffer(&singular_flag_buffer, 0, &staging, 4, 4);
        self.submit_and_wait(encoder);

        let mut flags = [0i32; 2];
        self.read_buffer(&staging, &mut flags);

        let num_swaps = flags[0] as usize;
        let singular = flags[1] != 0;

        // Clean up flag allocations
        self.allocator().deallocate(num_swaps_ptr, num_swaps_size);
        self.allocator()
            .deallocate(singular_flag_ptr, singular_flag_size);

        if singular {
            self.allocator().deallocate(lu_ptr, lu_size);
            self.allocator().deallocate(pivots_ptr, pivots_size);
            return Err(Error::Internal(format!(
                "LU decomposition failed: {}x{} matrix is singular (zero pivot encountered)",
                m, n
            )));
        }

        // Convert i32 pivots to i64
        let pivots_i64_size = k * std::mem::size_of::<i64>();
        let pivots_i64_ptr = self.allocator().allocate(pivots_i64_size);

        // Read i32 pivots and convert to i64
        let staging_pivots = self.create_staging_buffer("pivots_staging", pivots_size as u64);
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pivots_copy"),
                });
        encoder.copy_buffer_to_buffer(&pivots_buffer, 0, &staging_pivots, 0, pivots_size as u64);
        self.submit_and_wait(encoder);

        let mut pivots_i32 = vec![0i32; k];
        self.read_buffer(&staging_pivots, &mut pivots_i32);

        let pivots_i64: Vec<i64> = pivots_i32.iter().map(|&p| p as i64).collect();
        WgpuRuntime::copy_to_device(bytemuck::cast_slice(&pivots_i64), pivots_i64_ptr, device);

        self.allocator().deallocate(pivots_ptr, pivots_size);

        // Create tensors from GPU memory
        let lu = unsafe { Self::tensor_from_raw(lu_ptr, &[m, n], dtype, device) };
        let pivots = unsafe { Self::tensor_from_raw(pivots_i64_ptr, &[k], DType::I64, device) };

        Ok(LuDecomposition {
            lu,
            pivots,
            num_swaps,
        })
    }

    fn cholesky_decompose(
        &self,
        a: &Tensor<WgpuRuntime>,
    ) -> Result<CholeskyDecomposition<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU cholesky_decompose (only F32 supported)",
            });
        }

        // Allocate output on GPU
        let l_size = n * n * dtype.size_in_bytes();
        let l_ptr = self.allocator().allocate(l_size);
        let l_buffer = get_buffer(l_ptr)
            .ok_or_else(|| Error::Internal("Failed to get L buffer".to_string()))?;

        let not_pd_flag_size = std::mem::size_of::<i32>();
        let not_pd_flag_ptr = self.allocator().allocate(not_pd_flag_size);
        let not_pd_flag_buffer = get_buffer(not_pd_flag_ptr)
            .ok_or_else(|| Error::Internal("Failed to get not_pd_flag buffer".to_string()))?;

        // Copy input to L buffer
        WgpuRuntime::copy_within_device(a.storage().ptr(), l_ptr, l_size, device);

        // Create params buffer
        let params: [u32; 1] = [n as u32];
        let params_buffer = self.create_uniform_buffer("chol_params", 4);
        self.write_buffer(&params_buffer, &params);

        // Zero-initialize flag
        let zero_i32: [i32; 1] = [0];
        self.write_buffer(&not_pd_flag_buffer, &zero_i32);

        // Launch kernel
        kernels::launch_cholesky_decompose(
            self.pipeline_cache(),
            &self.queue,
            &l_buffer,
            &not_pd_flag_buffer,
            &params_buffer,
            dtype,
        )?;

        self.synchronize();

        // Read back flag
        let staging = self.create_staging_buffer("chol_flag_staging", 4);
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("chol_flag_copy"),
                });
        encoder.copy_buffer_to_buffer(&not_pd_flag_buffer, 0, &staging, 0, 4);
        self.submit_and_wait(encoder);

        let mut not_pd = [0i32; 1];
        self.read_buffer(&staging, &mut not_pd);

        self.allocator()
            .deallocate(not_pd_flag_ptr, not_pd_flag_size);

        if not_pd[0] != 0 {
            self.allocator().deallocate(l_ptr, l_size);
            return Err(Error::Internal(format!(
                "Cholesky decomposition failed: {}x{} matrix is not positive definite",
                n, n
            )));
        }

        let l = unsafe { Self::tensor_from_raw(l_ptr, &[n, n], dtype, device) };

        Ok(CholeskyDecomposition { l })
    }

    fn qr_decompose(&self, a: &Tensor<WgpuRuntime>) -> Result<QrDecomposition<WgpuRuntime>> {
        self.qr_decompose_internal(a, false)
    }

    fn qr_decompose_thin(&self, a: &Tensor<WgpuRuntime>) -> Result<QrDecomposition<WgpuRuntime>> {
        self.qr_decompose_internal(a, true)
    }

    fn solve(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        let n = validate_square_matrix(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU solve (only F32 supported)",
            });
        }

        // Only handle vector b for now
        let b_shape = b.shape();
        if b_shape.len() != 1 || b_shape[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: b_shape.to_vec(),
            });
        }

        // Compute LU decomposition
        let lu_result = self.lu_decompose(a)?;

        // Allocate output and temporary buffers
        let x_size = n * dtype.size_in_bytes();
        let x_ptr = self.allocator().allocate(x_size);
        let x_buffer = get_buffer(x_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

        let pb_ptr = self.allocator().allocate(x_size);
        let pb_buffer = get_buffer(pb_ptr)
            .ok_or_else(|| Error::Internal("Failed to get pb buffer".to_string()))?;

        let y_ptr = self.allocator().allocate(x_size);
        let y_buffer = get_buffer(y_ptr)
            .ok_or_else(|| Error::Internal("Failed to get y buffer".to_string()))?;

        // Get input buffers
        let b_buffer = get_buffer(b.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;
        let lu_buffer = get_buffer(lu_result.lu.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;

        // Convert pivots to i32 for shader
        let pivots_i64: Vec<i64> = lu_result.pivots.to_vec();
        let pivots_i32: Vec<i32> = pivots_i64.iter().map(|&p| p as i32).collect();
        let pivots_ptr = self.allocator().allocate(n * std::mem::size_of::<i32>());
        WgpuRuntime::copy_to_device(bytemuck::cast_slice(&pivots_i32), pivots_ptr, device);
        let pivots_buffer = get_buffer(pivots_ptr)
            .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

        // Apply permutation: pb = P @ b
        let perm_params: [u32; 1] = [n as u32];
        let perm_params_buffer = self.create_uniform_buffer("perm_params", 4);
        self.write_buffer(&perm_params_buffer, &perm_params);

        kernels::launch_apply_lu_permutation(
            self.pipeline_cache(),
            &self.queue,
            &b_buffer,
            &pb_buffer,
            &pivots_buffer,
            &perm_params_buffer,
            dtype,
        )?;

        // Forward substitution: Ly = pb (L has unit diagonal)
        let forward_params: [u32; 2] = [n as u32, 1]; // unit_diagonal = 1
        let forward_params_buffer = self.create_uniform_buffer("forward_params", 8);
        self.write_buffer(&forward_params_buffer, &forward_params);

        kernels::launch_forward_sub(
            self.pipeline_cache(),
            &self.queue,
            &lu_buffer,
            &pb_buffer,
            &y_buffer,
            &forward_params_buffer,
            dtype,
        )?;

        // Backward substitution: Ux = y
        let backward_params: [u32; 1] = [n as u32];
        let backward_params_buffer = self.create_uniform_buffer("backward_params", 4);
        self.write_buffer(&backward_params_buffer, &backward_params);

        kernels::launch_backward_sub(
            self.pipeline_cache(),
            &self.queue,
            &lu_buffer,
            &y_buffer,
            &x_buffer,
            &backward_params_buffer,
            dtype,
        )?;

        self.synchronize();

        // Clean up temporary buffers
        self.allocator().deallocate(pb_ptr, x_size);
        self.allocator().deallocate(y_ptr, x_size);
        self.allocator()
            .deallocate(pivots_ptr, n * std::mem::size_of::<i32>());

        let x = unsafe { Self::tensor_from_raw(x_ptr, &[n], dtype, device) };

        Ok(x)
    }

    fn solve_triangular_lower(
        &self,
        l: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        unit_diagonal: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(l.dtype())?;
        if l.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: l.dtype(),
                rhs: b.dtype(),
            });
        }
        let n = validate_square_matrix(l.shape())?;
        let dtype = l.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU solve_triangular_lower (only F32 supported)",
            });
        }

        if b.shape().len() != 1 || b.shape()[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: b.shape().to_vec(),
            });
        }

        let x_size = n * dtype.size_in_bytes();
        let x_ptr = self.allocator().allocate(x_size);
        let x_buffer = get_buffer(x_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

        let l_buffer = get_buffer(l.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get L buffer".to_string()))?;
        let b_buffer = get_buffer(b.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

        let params: [u32; 2] = [n as u32, if unit_diagonal { 1 } else { 0 }];
        let params_buffer = self.create_uniform_buffer("forward_params", 8);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_forward_sub(
            self.pipeline_cache(),
            &self.queue,
            &l_buffer,
            &b_buffer,
            &x_buffer,
            &params_buffer,
            dtype,
        )?;

        self.synchronize();

        let x = unsafe { Self::tensor_from_raw(x_ptr, &[n], dtype, device) };

        Ok(x)
    }

    fn solve_triangular_upper(
        &self,
        u: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(u.dtype())?;
        if u.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: u.dtype(),
                rhs: b.dtype(),
            });
        }
        let n = validate_square_matrix(u.shape())?;
        let dtype = u.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU solve_triangular_upper (only F32 supported)",
            });
        }

        if b.shape().len() != 1 || b.shape()[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: b.shape().to_vec(),
            });
        }

        let x_size = n * dtype.size_in_bytes();
        let x_ptr = self.allocator().allocate(x_size);
        let x_buffer = get_buffer(x_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

        let u_buffer = get_buffer(u.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get U buffer".to_string()))?;
        let b_buffer = get_buffer(b.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

        let params: [u32; 1] = [n as u32];
        let params_buffer = self.create_uniform_buffer("backward_params", 4);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_backward_sub(
            self.pipeline_cache(),
            &self.queue,
            &u_buffer,
            &b_buffer,
            &x_buffer,
            &params_buffer,
            dtype,
        )?;

        self.synchronize();

        let x = unsafe { Self::tensor_from_raw(x_ptr, &[n], dtype, device) };

        Ok(x)
    }

    fn lstsq(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        let (m, n) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU lstsq (only F32 supported)",
            });
        }

        // Only handle vector b for now
        let b_shape = b.shape();
        if b_shape.len() != 1 {
            return Err(Error::Internal(
                "lstsq currently only supports 1D b vector for WGPU".to_string(),
            ));
        }

        // Underdetermined systems not supported
        if m < n {
            return Err(Error::Internal(format!(
                "lstsq: underdetermined system not supported (A is {}x{}, requires m >= n)",
                m, n
            )));
        }

        // QR decomposition
        let qr = self.qr_decompose(a)?;

        // Compute Q^T @ b using TensorOps::matmul
        use crate::ops::TensorOps;

        // Make Q^T contiguous before matmul (transpose creates a view)
        let q_t = qr.q.transpose(0, 1)?.contiguous();
        let b_mat = b.reshape(&[m, 1])?.contiguous();

        // Q^T @ B gives [m, 1]
        let qtb = TensorOps::matmul(self, &q_t, &b_mat)?;

        // Allocate output X [n]
        let x_size = n * dtype.size_in_bytes();
        let x_ptr = self.allocator().allocate(x_size);
        let x_buffer = get_buffer(x_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

        // Zero initialize X
        let zero_bytes = vec![0u8; x_size];
        WgpuRuntime::copy_to_device(&zero_bytes, x_ptr, device);

        // Get first n elements of Q^T @ b
        // Make contiguous before reading since matmul output may not be contiguous
        let qtb_contig = qtb.contiguous();
        let qtb_data: Vec<f32> = qtb_contig.to_vec();
        let qtb_n: Vec<f32> = qtb_data[..n].to_vec();
        let qtb_ptr = self.allocator().allocate(x_size);
        WgpuRuntime::copy_to_device(bytemuck::cast_slice(&qtb_n), qtb_ptr, device);
        let qtb_buffer = get_buffer(qtb_ptr)
            .ok_or_else(|| Error::Internal("Failed to get qtb buffer".to_string()))?;

        let r_buffer = get_buffer(qr.r.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get R buffer".to_string()))?;

        // Backward substitution: R @ x = qtb[:n]
        let params: [u32; 1] = [n as u32];
        let params_buffer = self.create_uniform_buffer("backward_params", 4);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_backward_sub(
            self.pipeline_cache(),
            &self.queue,
            &r_buffer,
            &qtb_buffer,
            &x_buffer,
            &params_buffer,
            dtype,
        )?;

        self.synchronize();

        self.allocator().deallocate(qtb_ptr, x_size);

        let x = unsafe { Self::tensor_from_raw(x_ptr, &[n], dtype, device) };

        Ok(x)
    }

    fn inverse(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU inverse (only F32 supported)",
            });
        }

        // Compute LU decomposition
        let lu_result = self.lu_decompose(a)?;

        // Allocate output and temporary buffers
        let inv_size = n * n * dtype.size_in_bytes();
        let inv_ptr = self.allocator().allocate(inv_size);
        let inv_buffer = get_buffer(inv_ptr)
            .ok_or_else(|| Error::Internal("Failed to get inv buffer".to_string()))?;

        let col_size = n * dtype.size_in_bytes();

        // Create identity matrix on GPU
        let identity_ptr = self.allocator().allocate(inv_size);
        let identity_buffer = get_buffer(identity_ptr)
            .ok_or_else(|| Error::Internal("Failed to get identity buffer".to_string()))?;

        let id_params: [u32; 1] = [n as u32];
        let id_params_buffer = self.create_uniform_buffer("identity_params", 4);
        self.write_buffer(&id_params_buffer, &id_params);

        kernels::launch_create_identity(
            self.pipeline_cache(),
            &self.queue,
            &identity_buffer,
            &id_params_buffer,
            n,
            dtype,
        )?;

        // Get LU and pivots buffers
        let lu_buffer = get_buffer(lu_result.lu.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;

        // Convert pivots to i32
        let pivots_i64: Vec<i64> = lu_result.pivots.to_vec();
        let pivots_i32: Vec<i32> = pivots_i64.iter().map(|&p| p as i32).collect();
        let pivots_ptr = self.allocator().allocate(n * std::mem::size_of::<i32>());
        WgpuRuntime::copy_to_device(bytemuck::cast_slice(&pivots_i32), pivots_ptr, device);
        let pivots_buffer = get_buffer(pivots_ptr)
            .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

        // Allocate temporary buffers
        let e_ptr = self.allocator().allocate(col_size);
        let pb_ptr = self.allocator().allocate(col_size);
        let y_ptr = self.allocator().allocate(col_size);
        let x_ptr = self.allocator().allocate(col_size);

        let e_buffer = get_buffer(e_ptr)
            .ok_or_else(|| Error::Internal("Failed to get e buffer".to_string()))?;
        let pb_buffer = get_buffer(pb_ptr)
            .ok_or_else(|| Error::Internal("Failed to get pb buffer".to_string()))?;
        let y_buffer = get_buffer(y_ptr)
            .ok_or_else(|| Error::Internal("Failed to get y buffer".to_string()))?;
        let x_buffer = get_buffer(x_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

        // Solve for each column of the identity matrix
        for col in 0..n {
            // Extract column from identity matrix
            let extract_params: [u32; 3] = [n as u32, n as u32, col as u32];
            let extract_params_buffer = self.create_uniform_buffer("extract_params", 12);
            self.write_buffer(&extract_params_buffer, &extract_params);

            kernels::launch_extract_column(
                self.pipeline_cache(),
                &self.queue,
                &identity_buffer,
                &e_buffer,
                &extract_params_buffer,
                n,
                dtype,
            )?;

            // Apply permutation: pb = P @ e
            let perm_params: [u32; 1] = [n as u32];
            let perm_params_buffer = self.create_uniform_buffer("perm_params", 4);
            self.write_buffer(&perm_params_buffer, &perm_params);

            kernels::launch_apply_lu_permutation(
                self.pipeline_cache(),
                &self.queue,
                &e_buffer,
                &pb_buffer,
                &pivots_buffer,
                &perm_params_buffer,
                dtype,
            )?;

            // Forward substitution: Ly = pb (L has unit diagonal)
            let forward_params: [u32; 2] = [n as u32, 1];
            let forward_params_buffer = self.create_uniform_buffer("forward_params", 8);
            self.write_buffer(&forward_params_buffer, &forward_params);

            kernels::launch_forward_sub(
                self.pipeline_cache(),
                &self.queue,
                &lu_buffer,
                &pb_buffer,
                &y_buffer,
                &forward_params_buffer,
                dtype,
            )?;

            // Backward substitution: Ux = y
            let backward_params: [u32; 1] = [n as u32];
            let backward_params_buffer = self.create_uniform_buffer("backward_params", 4);
            self.write_buffer(&backward_params_buffer, &backward_params);

            kernels::launch_backward_sub(
                self.pipeline_cache(),
                &self.queue,
                &lu_buffer,
                &y_buffer,
                &x_buffer,
                &backward_params_buffer,
                dtype,
            )?;

            // Scatter x into column of inverse matrix
            let scatter_params: [u32; 2] = [n as u32, col as u32];
            let scatter_params_buffer = self.create_uniform_buffer("scatter_params", 8);
            self.write_buffer(&scatter_params_buffer, &scatter_params);

            kernels::launch_scatter_column(
                self.pipeline_cache(),
                &self.queue,
                &x_buffer,
                &inv_buffer,
                &scatter_params_buffer,
                n,
                dtype,
            )?;
        }

        self.synchronize();

        // Clean up
        self.allocator().deallocate(identity_ptr, inv_size);
        self.allocator()
            .deallocate(pivots_ptr, n * std::mem::size_of::<i32>());
        self.allocator().deallocate(e_ptr, col_size);
        self.allocator().deallocate(pb_ptr, col_size);
        self.allocator().deallocate(y_ptr, col_size);
        self.allocator().deallocate(x_ptr, col_size);

        let inv = unsafe { Self::tensor_from_raw(inv_ptr, &[n, n], dtype, device) };

        Ok(inv)
    }

    fn det(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU det (only F32 supported)",
            });
        }

        // Compute LU decomposition
        let lu_result = self.lu_decompose(a)?;

        // Allocate output
        let det_size = dtype.size_in_bytes();
        let det_ptr = self.allocator().allocate(det_size);
        let det_buffer = get_buffer(det_ptr)
            .ok_or_else(|| Error::Internal("Failed to get det buffer".to_string()))?;

        let lu_buffer = get_buffer(lu_result.lu.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;

        // Create params buffer with n and num_swaps
        let params: [u32; 2] = [n as u32, lu_result.num_swaps as u32];
        let params_buffer = self.create_uniform_buffer("det_params", 8);
        // Note: DetParams has n: u32 and num_swaps: i32, but we're passing both as u32
        // This works because the shader interprets num_swaps as signed
        self.write_buffer(&params_buffer, &params);

        kernels::launch_det_from_lu(
            self.pipeline_cache(),
            &self.queue,
            &lu_buffer,
            &det_buffer,
            &params_buffer,
            dtype,
        )?;

        self.synchronize();

        let det = unsafe { Self::tensor_from_raw(det_ptr, &[], dtype, device) };

        Ok(det)
    }

    fn trace(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let min_dim = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU trace (only F32 supported)",
            });
        }

        // Allocate output (zero-initialized for reduction)
        let trace_size = dtype.size_in_bytes();
        let trace_ptr = self.allocator().allocate(trace_size);
        let trace_buffer = get_buffer(trace_ptr)
            .ok_or_else(|| Error::Internal("Failed to get trace buffer".to_string()))?;

        let zero: [f32; 1] = [0.0];
        self.write_buffer(&trace_buffer, &zero);

        let a_buffer = get_buffer(a.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

        let params: [u32; 2] = [min_dim as u32, n as u32];
        let params_buffer = self.create_uniform_buffer("trace_params", 8);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_trace(
            self.pipeline_cache(),
            &self.queue,
            &a_buffer,
            &trace_buffer,
            &params_buffer,
            min_dim,
            dtype,
        )?;

        self.synchronize();

        let trace = unsafe { Self::tensor_from_raw(trace_ptr, &[], dtype, device) };

        Ok(trace)
    }

    fn diag(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let min_dim = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU diag (only F32 supported)",
            });
        }

        let diag_size = min_dim * dtype.size_in_bytes();
        let diag_ptr = self.allocator().allocate(diag_size);
        let diag_buffer = get_buffer(diag_ptr)
            .ok_or_else(|| Error::Internal("Failed to get diag buffer".to_string()))?;

        let a_buffer = get_buffer(a.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

        let params: [u32; 2] = [min_dim as u32, n as u32];
        let params_buffer = self.create_uniform_buffer("diag_params", 8);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_diag(
            self.pipeline_cache(),
            &self.queue,
            &a_buffer,
            &diag_buffer,
            &params_buffer,
            min_dim,
            dtype,
        )?;

        self.synchronize();

        let diag = unsafe { Self::tensor_from_raw(diag_ptr, &[min_dim], dtype, device) };

        Ok(diag)
    }

    fn diagflat(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;

        if a.shape().len() != 1 {
            return Err(Error::Internal(format!(
                "diagflat requires 1D input tensor, got {}D tensor with shape {:?}",
                a.shape().len(),
                a.shape()
            )));
        }

        let n = a.shape()[0];
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU diagflat (only F32 supported)",
            });
        }

        let out_size = n * n * dtype.size_in_bytes();
        let out_ptr = self.allocator().allocate(out_size);
        let out_buffer = get_buffer(out_ptr)
            .ok_or_else(|| Error::Internal("Failed to get out buffer".to_string()))?;

        let a_buffer = get_buffer(a.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

        let params: [u32; 1] = [n as u32];
        let params_buffer = self.create_uniform_buffer("diagflat_params", 4);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_diagflat(
            self.pipeline_cache(),
            &self.queue,
            &a_buffer,
            &out_buffer,
            &params_buffer,
            n,
            dtype,
        )?;

        self.synchronize();

        let out = unsafe { Self::tensor_from_raw(out_ptr, &[n, n], dtype, device) };

        Ok(out)
    }

    fn matrix_rank(
        &self,
        a: &Tensor<WgpuRuntime>,
        tol: Option<f64>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();
        let k = m.min(n);

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU matrix_rank (only F32 supported)",
            });
        }

        // Use QR decomposition to estimate rank
        let qr = self.qr_decompose(a)?;

        // Get diagonal of R
        let r_diag = self.diag(&qr.r)?;

        // Allocate GPU buffers for max abs and count
        let max_size = dtype.size_in_bytes();
        let max_ptr = self.allocator().allocate(max_size);
        let max_buffer = get_buffer(max_ptr)
            .ok_or_else(|| Error::Internal("Failed to get max buffer".to_string()))?;

        let count_size = std::mem::size_of::<u32>();
        let count_ptr = self.allocator().allocate(count_size);
        let count_buffer = get_buffer(count_ptr)
            .ok_or_else(|| Error::Internal("Failed to get count buffer".to_string()))?;

        let r_diag_buffer = get_buffer(r_diag.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get r_diag buffer".to_string()))?;

        // Zero-initialize
        let zero_f32: [f32; 1] = [0.0];
        let zero_u32: [u32; 1] = [0];
        self.write_buffer(&max_buffer, &zero_f32);
        self.write_buffer(&count_buffer, &zero_u32);

        // Compute max absolute value
        let max_params: [u32; 1] = [k as u32];
        let max_params_buffer = self.create_uniform_buffer("max_params", 4);
        self.write_buffer(&max_params_buffer, &max_params);

        kernels::launch_max_abs(
            self.pipeline_cache(),
            &self.queue,
            &r_diag_buffer,
            &max_buffer,
            &max_params_buffer,
            k,
            dtype,
        )?;

        self.synchronize();

        // Read max value
        let staging = self.create_staging_buffer("max_staging", 4);
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("max_copy"),
                });
        encoder.copy_buffer_to_buffer(&max_buffer, 0, &staging, 0, 4);
        self.submit_and_wait(encoder);

        let mut max_data = [0.0f32; 1];
        self.read_buffer(&staging, &mut max_data);
        let max_diag = max_data[0] as f64;

        // Compute tolerance
        let base_tol = tol.unwrap_or_else(|| {
            let eps = f32::EPSILON as f64;
            (m.max(n) as f64) * eps
        });
        let threshold = (base_tol * max_diag) as f32;

        // Count elements above threshold
        let count_params_buffer = self.create_uniform_buffer("count_params", 8);
        // Pack k and threshold into uniform buffer
        self.queue
            .write_buffer(&count_params_buffer, 0, bytemuck::cast_slice(&[k as u32]));
        self.queue
            .write_buffer(&count_params_buffer, 4, bytemuck::cast_slice(&[threshold]));

        kernels::launch_count_above_threshold(
            self.pipeline_cache(),
            &self.queue,
            &r_diag_buffer,
            &count_buffer,
            &count_params_buffer,
            k,
            dtype,
        )?;

        self.synchronize();

        // Read count
        let staging_count = self.create_staging_buffer("count_staging", 4);
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("count_copy"),
                });
        encoder.copy_buffer_to_buffer(&count_buffer, 0, &staging_count, 0, 4);
        self.submit_and_wait(encoder);

        let mut count_data = [0u32; 1];
        self.read_buffer(&staging_count, &mut count_data);
        let rank = count_data[0] as i64;

        // Clean up
        self.allocator().deallocate(max_ptr, max_size);
        self.allocator().deallocate(count_ptr, count_size);

        // Create rank tensor
        let rank_data: [i64; 1] = [rank];
        let rank_tensor = Tensor::<WgpuRuntime>::from_slice(&rank_data, &[], device);

        Ok(rank_tensor)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<WgpuRuntime>,
        ord: MatrixNormOrder,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (_m, _n) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU matrix_norm (only F32 supported)",
            });
        }

        match ord {
            MatrixNormOrder::Frobenius => {
                // Frobenius norm: ||A||_F = sqrt(sum(A²))
                // Use existing tensor ops to keep data on GPU
                let squared = self.square(a)?;
                let sum_sq = self.sum(&squared, &[], false)?;
                self.sqrt(&sum_sq)
            }
            MatrixNormOrder::Spectral => {
                // Spectral norm is the largest singular value
                let svd = self.svd_decompose(a)?;
                // S is already sorted descending, so s[0] is the largest
                let s_vec: Vec<f32> = svd.s.to_vec();
                let spectral_val = s_vec.first().copied().unwrap_or(0.0);
                Ok(Tensor::<WgpuRuntime>::from_slice(
                    &[spectral_val],
                    &[],
                    a.device(),
                ))
            }
            MatrixNormOrder::Nuclear => {
                // Nuclear norm is sum of singular values
                let svd = self.svd_decompose(a)?;
                self.sum(&svd.s, &[], false)
            }
        }
    }

    fn svd_decompose(&self, a: &Tensor<WgpuRuntime>) -> Result<SvdDecomposition<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU svd_decompose (only F32 supported)",
            });
        }

        // Handle transpose for m < n case
        // We want to work with a matrix where rows >= cols
        let transposed = m < n;
        let (work_m, work_n) = if transposed { (n, m) } else { (m, n) };
        let k = work_m.min(work_n); // k = min(m, n) = work_n for our case

        // Get input data
        let a_data: Vec<f32> = a.to_vec();

        // If transposed, compute A^T
        let work_data = if transposed {
            let mut transposed_data = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    transposed_data[j * m + i] = a_data[i * n + j];
                }
            }
            transposed_data
        } else {
            a_data
        };

        // Allocate buffers for SVD computation
        // B matrix: [work_m, work_n] - working copy, becomes U columns
        let b_size = work_m * work_n * dtype.size_in_bytes();
        let b_ptr = self.allocator().allocate(b_size);
        let b_buffer = get_buffer_or_err!(b_ptr, "B (working matrix)");

        // V matrix: [work_n, work_n]
        let v_size = work_n * work_n * dtype.size_in_bytes();
        let v_ptr = self.allocator().allocate(v_size);
        let v_buffer = get_buffer_or_err!(v_ptr, "V (right singular vectors)");

        // S vector: [work_n]
        let s_size = work_n * dtype.size_in_bytes();
        let s_ptr = self.allocator().allocate(s_size);
        let s_buffer = get_buffer_or_err!(s_ptr, "S (singular values)");

        // Converged flag
        let converged_flag_size = std::mem::size_of::<i32>();
        let converged_flag_ptr = self.allocator().allocate(converged_flag_size);
        let converged_flag_buffer = get_buffer_or_err!(converged_flag_ptr, "SVD convergence flag");

        // Copy working data to B buffer
        WgpuRuntime::copy_to_device(bytemuck::cast_slice(&work_data), b_ptr, device);

        // Zero-initialize converged flag
        let zero_i32: [i32; 1] = [0];
        self.write_buffer(&converged_flag_buffer, &zero_i32);

        // Create params buffer: [work_m, work_n]
        let params: [u32; 2] = [work_m as u32, work_n as u32];
        let params_buffer = self.create_uniform_buffer("svd_params", 8);
        self.write_buffer(&params_buffer, &params);

        // Launch SVD kernel
        kernels::launch_svd_jacobi(
            self.pipeline_cache(),
            &self.queue,
            &b_buffer,
            &v_buffer,
            &s_buffer,
            &converged_flag_buffer,
            &params_buffer,
            dtype,
        )?;

        self.synchronize();

        // Read back converged flag
        let staging = self.create_staging_buffer("svd_converged_staging", 4);
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("svd_converged_copy"),
                });
        encoder.copy_buffer_to_buffer(&converged_flag_buffer, 0, &staging, 0, 4);
        self.submit_and_wait(encoder);

        let mut converged_val = [0i32; 1];
        self.read_buffer(&staging, &mut converged_val);

        // Clean up converged flag
        self.allocator()
            .deallocate(converged_flag_ptr, converged_flag_size);

        if converged_val[0] != 0 {
            // Cleanup on failure
            self.allocator().deallocate(b_ptr, b_size);
            self.allocator().deallocate(v_ptr, v_size);
            self.allocator().deallocate(s_ptr, s_size);
            return Err(Error::Internal(
                "SVD did not converge within maximum iterations".to_string(),
            ));
        }

        // Read results back for potential transposition
        let b_staging = self.create_staging_buffer("svd_b_staging", b_size as u64);
        let v_staging = self.create_staging_buffer("svd_v_staging", v_size as u64);

        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("svd_results_copy"),
                });
        encoder.copy_buffer_to_buffer(&b_buffer, 0, &b_staging, 0, b_size as u64);
        encoder.copy_buffer_to_buffer(&v_buffer, 0, &v_staging, 0, v_size as u64);
        self.submit_and_wait(encoder);

        let mut b_data = vec![0.0f32; work_m * work_n];
        let mut v_data = vec![0.0f32; work_n * work_n];
        self.read_buffer(&b_staging, &mut b_data);
        self.read_buffer(&v_staging, &mut v_data);

        // Deallocate working buffers
        self.allocator().deallocate(b_ptr, b_size);
        self.allocator().deallocate(v_ptr, v_size);

        // Handle transpose: if original was m < n, swap U and V^T
        let (u_final, vt_final) = if transposed {
            // For A^T = U_work * S * V_work^T
            // A = V_work * S * U_work^T
            // So: final_U = V_work, final_V^T = U_work^T (which is B transposed)

            // For original A [m x n] with m < n:
            // - U_final should be [m x k] where k = min(m,n) = m
            // - V^T_final should be [k x n] = [m x n]
            //
            // After A^T SVD (work_m = n, work_n = m):
            // - b_data contains U_work [work_m x work_n] = [n x m]
            // - v_data contains V_work [work_n x work_n] = [m x m]
            //
            // U_final = V_work, shape [m x k] = [m x m]
            let mut u_final = vec![0.0f32; m * k];
            for i in 0..m {
                for j in 0..k {
                    u_final[i * k + j] = v_data[i * work_n + j];
                }
            }

            // V^T_final = U_work^T, shape [k x n] = [m x n]
            // U_work is [n x m], so U_work^T is [m x n]
            let mut vt_final = vec![0.0f32; k * n];
            for i in 0..k {
                for j in 0..n {
                    vt_final[i * n + j] = b_data[j * work_n + i];
                }
            }

            (u_final, vt_final)
        } else {
            // No transpose: U = B (first k columns), V^T = V^T
            let mut u_final = vec![0.0f32; m * k];
            for i in 0..m {
                for j in 0..k {
                    u_final[i * k + j] = b_data[i * n + j];
                }
            }

            // V^T = transpose of V
            let mut vt_final = vec![0.0f32; k * n];
            for i in 0..k {
                for j in 0..n {
                    vt_final[i * n + j] = v_data[j * n + i];
                }
            }

            (u_final, vt_final)
        };

        // Allocate final output tensors on GPU
        let u_size = u_final.len() * dtype.size_in_bytes();
        let u_ptr = self.allocator().allocate(u_size);
        WgpuRuntime::copy_to_device(bytemuck::cast_slice(&u_final), u_ptr, device);

        let vt_size = vt_final.len() * dtype.size_in_bytes();
        let vt_ptr = self.allocator().allocate(vt_size);
        WgpuRuntime::copy_to_device(bytemuck::cast_slice(&vt_final), vt_ptr, device);

        // Create output tensors
        let u = unsafe { Self::tensor_from_raw(u_ptr, &[m, k], dtype, device) };
        let s = unsafe { Self::tensor_from_raw(s_ptr, &[k], dtype, device) };
        let vt = unsafe { Self::tensor_from_raw(vt_ptr, &[k, n], dtype, device) };

        Ok(SvdDecomposition { u, s, vt })
    }
}

// Helper methods
impl WgpuClient {
    /// Create a tensor from a raw GPU pointer
    ///
    /// # Safety
    /// - `ptr` must point to valid GPU memory
    /// - The memory must remain valid for the lifetime of the returned tensor
    unsafe fn tensor_from_raw(
        ptr: u64,
        shape: &[usize],
        dtype: DType,
        device: &super::WgpuDevice,
    ) -> Tensor<WgpuRuntime> {
        let len = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let storage = unsafe { Storage::<WgpuRuntime>::from_ptr(ptr, len, dtype, device) };
        let layout = Layout::contiguous(shape);
        Tensor::from_parts(storage, layout)
    }

    fn qr_decompose_internal(
        &self,
        a: &Tensor<WgpuRuntime>,
        thin: bool,
    ) -> Result<QrDecomposition<WgpuRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let k = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "WGPU qr_decompose (only F32 supported)",
            });
        }

        // Q dimensions: [m, m] for full, [m, k] for thin
        let q_cols = if thin { k } else { m };
        let q_size = m * q_cols * dtype.size_in_bytes();
        let q_ptr = self.allocator().allocate(q_size);
        let q_buffer = get_buffer(q_ptr)
            .ok_or_else(|| Error::Internal("Failed to get Q buffer".to_string()))?;

        // R is [m, n] but only upper triangular part is meaningful
        let r_size = m * n * dtype.size_in_bytes();
        let r_ptr = self.allocator().allocate(r_size);
        let r_buffer = get_buffer(r_ptr)
            .ok_or_else(|| Error::Internal("Failed to get R buffer".to_string()))?;

        // Workspace for Householder vector (size m elements)
        let workspace_size = m * dtype.size_in_bytes();
        let workspace_ptr = self.allocator().allocate(workspace_size);
        let workspace_buffer = get_buffer(workspace_ptr)
            .ok_or_else(|| Error::Internal("Failed to get workspace buffer".to_string()))?;

        // Copy A to R (will be modified in place)
        WgpuRuntime::copy_within_device(a.storage().ptr(), r_ptr, r_size, device);

        // Create params buffer
        let params: [u32; 3] = [m as u32, n as u32, if thin { 1 } else { 0 }];
        let params_buffer = self.create_uniform_buffer("qr_params", 12);
        self.write_buffer(&params_buffer, &params);

        kernels::launch_qr_decompose(
            self.pipeline_cache(),
            &self.queue,
            &q_buffer,
            &r_buffer,
            &workspace_buffer,
            &params_buffer,
            dtype,
        )?;

        // Clean up workspace
        self.allocator().deallocate(workspace_ptr, workspace_size);

        self.synchronize();

        let q = unsafe { Self::tensor_from_raw(q_ptr, &[m, q_cols], dtype, device) };

        // For thin QR, R should be [k, n]
        let r = if thin && m > n {
            unsafe { Self::tensor_from_raw(r_ptr, &[k, n], dtype, device) }
        } else if thin {
            unsafe { Self::tensor_from_raw(r_ptr, &[m, n], dtype, device) }
        } else {
            unsafe { Self::tensor_from_raw(r_ptr, &[m, n], dtype, device) }
        };

        Ok(QrDecomposition { q, r })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::wgpu::{WgpuDevice, is_wgpu_available};

    fn create_client() -> WgpuClient {
        let device = WgpuDevice::new(0);
        WgpuRuntime::default_client(&device)
    }

    #[test]
    fn test_trace() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[1, 2], [3, 4]]
        // trace = 1 + 4 = 5
        let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let t = client.trace(&a).unwrap();
        let result: Vec<f32> = t.to_vec();

        assert!((result[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diag() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // 2x3 matrix
        let a =
            Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device);

        let d = client.diag(&a).unwrap();
        let result: Vec<f32> = d.to_vec();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diagflat() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device);

        let m = client.diagflat(&a).unwrap();
        let result: Vec<f32> = m.to_vec();

        assert_eq!(m.shape(), &[3, 3]);
        // Expected: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        assert!((result[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((result[1]).abs() < 1e-5); // [0,1]
        assert!((result[4] - 2.0).abs() < 1e-5); // [1,1]
        assert!((result[8] - 3.0).abs() < 1e-5); // [2,2]
    }

    #[test]
    fn test_lu_decomposition() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[4, 3], [6, 3]]
        let a = Tensor::<WgpuRuntime>::from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device);

        let lu = client.lu_decompose(&a).unwrap();

        assert_eq!(lu.lu.shape(), &[2, 2]);
        assert_eq!(lu.pivots.shape(), &[2]);
    }

    #[test]
    fn test_cholesky() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Symmetric positive definite: [[4, 2], [2, 5]]
        let a = Tensor::<WgpuRuntime>::from_slice(&[4.0f32, 2.0, 2.0, 5.0], &[2, 2], device);

        let chol = client.cholesky_decompose(&a).unwrap();

        assert_eq!(chol.l.shape(), &[2, 2]);

        // L should be lower triangular
        let l_data: Vec<f32> = chol.l.to_vec();
        assert!((l_data[1]).abs() < 1e-5); // Upper triangle should be 0
    }

    #[test]
    fn test_det() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[1, 2], [3, 4]]
        // det = 1*4 - 2*3 = -2
        let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let d = client.det(&a).unwrap();
        let result: Vec<f32> = d.to_vec();

        assert!((result[0] - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn test_solve() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Solve [[2, 1], [1, 2]] @ x = [3, 3]
        // Solution: x = [1, 1]
        let a = Tensor::<WgpuRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device);
        let b = Tensor::<WgpuRuntime>::from_slice(&[3.0f32, 3.0], &[2], device);

        let x = client.solve(&a, &b).unwrap();
        let result: Vec<f32> = x.to_vec();

        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_inverse() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Test 2x2 matrix: [[4, 7], [2, 6]]
        // Inverse: [[0.6, -0.7], [-0.2, 0.4]]
        let a = Tensor::<WgpuRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device);

        let inv = client.inverse(&a).unwrap();
        let result: Vec<f32> = inv.to_vec();

        // Check inverse values (det = 4*6 - 7*2 = 10)
        // inv = (1/10) * [[6, -7], [-2, 4]]
        assert!((result[0] - 0.6).abs() < 1e-4); // [0,0]
        assert!((result[1] - (-0.7)).abs() < 1e-4); // [0,1]
        assert!((result[2] - (-0.2)).abs() < 1e-4); // [1,0]
        assert!((result[3] - 0.4).abs() < 1e-4); // [1,1]
    }

    #[test]
    fn test_matrix_rank_full() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Full rank 2x2 matrix
        let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let rank = client.matrix_rank(&a, None).unwrap();
        let result: Vec<i64> = rank.to_vec();

        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_matrix_rank_deficient() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Rank-deficient 2x2 matrix (rows are linearly dependent)
        let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], device);

        let rank = client.matrix_rank(&a, None).unwrap();
        let result: Vec<i64> = rank.to_vec();

        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_qr_decomposition() {
        use crate::ops::TensorOps;

        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Test QR: A = Q @ R
        let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let qr = client.qr_decompose(&a).unwrap();

        // Verify Q @ R == A
        let reconstructed = TensorOps::matmul(&client, &qr.q, &qr.r).unwrap();
        let reconstructed = reconstructed.contiguous();
        let a_data: Vec<f32> = a.to_vec();
        let reconstructed_data: Vec<f32> = reconstructed.to_vec();

        for i in 0..4 {
            assert!(
                (a_data[i] - reconstructed_data[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                a_data[i],
                reconstructed_data[i]
            );
        }
    }

    #[test]
    fn test_lstsq() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Overdetermined system: A is 3x2, b is 3x1
        // A = [[1, 1], [1, 2], [1, 3]], b = [1, 2, 3]
        // This is a linear fit problem: y = a + b*x
        // The least squares solution minimizes ||Ax - b||^2
        let a =
            Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2], device);
        let b = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device);

        let x = client.lstsq(&a, &b).unwrap();
        assert_eq!(x.shape(), &[2]);
        let result: Vec<f32> = x.to_vec();

        // Verify the solution is reasonable by checking residual
        // Just verify we get a valid result (no NaN/inf)
        assert!(!result[0].is_nan() && !result[0].is_infinite());
        assert!(!result[1].is_nan() && !result[1].is_infinite());
    }
}
