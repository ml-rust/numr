//! CPU runtime implementation
//!
//! The CPU runtime uses standard heap allocation and provides a reference
//! implementation for all tensor operations.
//!
//! # Broadcasting
//!
//! NumPy-style broadcasting is fully supported for binary arithmetic operations
//! (add, sub, mul, div, pow, max, min). Shapes are broadcast according to standard
//! rules: dimensions are right-aligned and expanded where one operand has size 1.
//!
//! Comparison operations (eq, ne, lt, le, gt, ge) also support broadcasting.
//!
//! # Non-contiguous Tensors
//!
//! Operations handle non-contiguous tensors via strided memory access. For
//! broadcasting, a strided kernel is used that correctly handles stride-0
//! dimensions (where a single value is broadcast across the dimension).

mod helpers;
mod kernels;
mod ops;

use super::{DefaultAllocator, Device, Runtime, RuntimeClient};
use crate::dtype::Element;
use crate::ops::{BinaryOp, Kernel, ReduceOp, UnaryOp};
use std::alloc::{Layout as AllocLayout, alloc_zeroed, dealloc};

// Re-export TensorOps, ScalarOps, CompareOps traits for convenience
pub use crate::ops::{CompareOps, ScalarOps, TensorOps};

// Re-export Tensor for tests
pub use crate::tensor::Tensor;

/// CPU compute runtime
///
/// This is the default runtime that works on any platform.
/// Memory is allocated on the heap using the system allocator.
#[derive(Clone, Debug, Default)]
pub struct CpuRuntime;

/// CPU device (there's only one: the host CPU)
#[derive(Clone, Debug, Default)]
pub struct CpuDevice {
    id: usize,
}

impl CpuDevice {
    /// Create a new CPU device
    pub fn new() -> Self {
        Self { id: 0 }
    }
}

impl Device for CpuDevice {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> String {
        "cpu".to_string()
    }
}

/// CPU client for operation dispatch
#[derive(Clone, Debug)]
pub struct CpuClient {
    pub(crate) device: CpuDevice,
    allocator: CpuAllocator,
}

impl CpuClient {
    /// Create a new CPU client
    pub fn new(device: CpuDevice) -> Self {
        let allocator = create_cpu_allocator(device.clone());
        Self { device, allocator }
    }
}

impl RuntimeClient<CpuRuntime> for CpuClient {
    fn device(&self) -> &CpuDevice {
        &self.device
    }

    fn synchronize(&self) {
        // CPU operations are synchronous, nothing to do
    }

    fn allocator(&self) -> &CpuAllocator {
        &self.allocator
    }
}

/// CPU-specific allocator type alias
pub type CpuAllocator = DefaultAllocator<CpuDevice>;

/// Create a CPU allocator for the given device
fn create_cpu_allocator(device: CpuDevice) -> CpuAllocator {
    DefaultAllocator::new(
        device,
        |size, _dev| {
            if size == 0 {
                return 0;
            }
            let align = 64; // AVX-512 alignment
            let layout =
                AllocLayout::from_size_align(size, align).expect("Invalid allocation layout");
            let ptr = unsafe { alloc_zeroed(layout) };
            if ptr.is_null() {
                panic!("Failed to allocate {} bytes", size);
            }
            ptr as u64
        },
        |ptr, size, _dev| {
            if ptr == 0 || size == 0 {
                return;
            }
            let align = 64;
            let layout =
                AllocLayout::from_size_align(size, align).expect("Invalid allocation layout");
            unsafe {
                dealloc(ptr as *mut u8, layout);
            }
        },
    )
}

impl Runtime for CpuRuntime {
    type Device = CpuDevice;
    type Client = CpuClient;
    type Allocator = CpuAllocator;
    type RawHandle = (); // CPU has no special handle needed

    fn name() -> &'static str {
        "cpu"
    }

    fn allocate(size_bytes: usize, _device: &Self::Device) -> u64 {
        if size_bytes == 0 {
            return 0;
        }

        // Use aligned allocation for SIMD compatibility
        let align = 64; // AVX-512 alignment
        let layout =
            AllocLayout::from_size_align(size_bytes, align).expect("Invalid allocation layout");

        let ptr = unsafe { alloc_zeroed(layout) };

        if ptr.is_null() {
            panic!("Failed to allocate {} bytes", size_bytes);
        }

        ptr as u64
    }

    fn deallocate(ptr: u64, size_bytes: usize, _device: &Self::Device) {
        if ptr == 0 || size_bytes == 0 {
            return;
        }

        let align = 64;
        let layout =
            AllocLayout::from_size_align(size_bytes, align).expect("Invalid allocation layout");

        unsafe {
            dealloc(ptr as *mut u8, layout);
        }
    }

    fn copy_to_device(src: &[u8], dst: u64, _device: &Self::Device) {
        if src.is_empty() || dst == 0 {
            return;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
        }
    }

    fn copy_from_device(src: u64, dst: &mut [u8], _device: &Self::Device) {
        if dst.is_empty() || src == 0 {
            return;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst.as_mut_ptr(), dst.len());
        }
    }

    fn copy_within_device(src: u64, dst: u64, size_bytes: usize, _device: &Self::Device) {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return;
        }

        unsafe {
            // Use copy (not copy_nonoverlapping) in case src and dst overlap
            std::ptr::copy(src as *const u8, dst as *mut u8, size_bytes);
        }
    }

    fn default_device() -> Self::Device {
        CpuDevice::new()
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        CpuClient::new(device.clone())
    }

    fn raw_handle(_client: &Self::Client) -> &Self::RawHandle {
        &()
    }
}

// ============================================================================
// Kernel Trait Implementation
// ============================================================================

#[allow(unsafe_op_in_unsafe_fn)] // Kernels are already marked unsafe
impl Kernel<CpuRuntime> for CpuClient {
    unsafe fn binary_op<T: Element>(
        &self,
        op: BinaryOp,
        a: *const T,
        b: *const T,
        out: *mut T,
        len: usize,
    ) {
        kernels::binary_op_kernel(op, a, b, out, len);
    }

    unsafe fn unary_op<T: Element>(&self, op: UnaryOp, a: *const T, out: *mut T, len: usize) {
        kernels::unary_op_kernel(op, a, out, len);
    }

    unsafe fn matmul<T: Element>(
        &self,
        a: *const T,
        b: *const T,
        out: *mut T,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    ) {
        kernels::matmul_kernel(a, b, out, m, n, k, lda, ldb, ldc);
    }

    unsafe fn reduce<T: Element>(
        &self,
        op: ReduceOp,
        a: *const T,
        out: *mut T,
        reduce_size: usize,
        outer_size: usize,
    ) {
        kernels::reduce_kernel(op, a, out, reduce_size, outer_size);
    }

    unsafe fn fill<T: Element>(&self, out: *mut T, value: T, len: usize) {
        kernels::fill_kernel(out, value, len);
    }

    unsafe fn copy<T: Element>(&self, src: *const T, dst: *mut T, len: usize) {
        kernels::copy_kernel(src, dst, len);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Allocator;

    #[test]
    fn test_allocate_deallocate() {
        let device = CpuDevice::new();
        let ptr = CpuRuntime::allocate(1024, &device);
        assert_ne!(ptr, 0);
        CpuRuntime::deallocate(ptr, 1024, &device);
    }

    #[test]
    fn test_copy_roundtrip() {
        let device = CpuDevice::new();
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let ptr = CpuRuntime::allocate(data.len(), &device);
        CpuRuntime::copy_to_device(&data, ptr, &device);

        let mut result = vec![0u8; data.len()];
        CpuRuntime::copy_from_device(ptr, &mut result, &device);

        assert_eq!(data, result);

        CpuRuntime::deallocate(ptr, data.len(), &device);
    }

    #[test]
    fn test_copy_within_device() {
        let device = CpuDevice::new();
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let src = CpuRuntime::allocate(data.len(), &device);
        let dst = CpuRuntime::allocate(data.len(), &device);

        CpuRuntime::copy_to_device(&data, src, &device);
        CpuRuntime::copy_within_device(src, dst, data.len(), &device);

        let mut result = vec![0u8; data.len()];
        CpuRuntime::copy_from_device(dst, &mut result, &device);

        assert_eq!(data, result);

        CpuRuntime::deallocate(src, data.len(), &device);
        CpuRuntime::deallocate(dst, data.len(), &device);
    }

    #[test]
    fn test_zero_allocation() {
        let device = CpuDevice::new();
        let ptr = CpuRuntime::allocate(0, &device);
        assert_eq!(ptr, 0);
        CpuRuntime::deallocate(ptr, 0, &device); // Should not panic
    }

    #[test]
    fn test_client_allocator() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let ptr = client.allocator().allocate(256);
        assert_ne!(ptr, 0);
        client.allocator().deallocate(ptr, 256);
    }

    #[test]
    fn test_raw_handle() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let _handle: &() = CpuRuntime::raw_handle(&client);
        // For CPU, handle is just ()
    }

    // ===== TensorOps Integration Tests =====

    #[test]
    fn test_tensor_add() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_tensor_sub() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);

        let c = client.sub(&a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_tensor_mul() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0, 5.0], &[4], &device);

        let c = client.mul(&a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_tensor_div() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 5.0, 8.0], &[4], &device);

        let c = client.div(&a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_tensor_neg() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -2.0, 3.0, -4.0], &[4], &device);

        let b = client.neg(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn test_tensor_sqrt() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 4.0, 9.0, 16.0], &[4], &device);

        let b = client.sqrt(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_exp() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device);

        let b = client.exp(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert!((result[0] - 1.0).abs() < 1e-6); // e^0 = 1
        // e^1 should be in range (2.7, 2.72)
        assert!(result[1] > 2.7 && result[1] < 2.72); // e^1 = e ≈ 2.718
    }

    #[test]
    fn test_tensor_matmul_2x2() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

        let c = TensorOps::matmul(&client, &a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_tensor_matmul_3x2_2x4() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[1, 2], [3, 4], [5, 6]] (3x2)
        // B = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4)
        // C = A @ B (3x4)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        );

        let c = TensorOps::matmul(&client, &a, &b).unwrap();

        assert_eq!(c.shape(), &[3, 4]);
        let result: Vec<f32> = c.to_vec();
        // Row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
        // Row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
        // Row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
        assert_eq!(
            result,
            [
                11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
            ]
        );
    }

    #[test]
    fn test_tensor_sum_last_dim() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Shape [2, 3] -> sum over dim 1 -> shape [2]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

        let b = client.sum(&a, &[1], false).unwrap();

        assert_eq!(b.shape(), &[2]);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_tensor_mean_last_dim() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0],
            &[2, 3],
            &device,
        );

        let b = client.mean(&a, &[1], false).unwrap();

        assert_eq!(b.shape(), &[2]);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [2.0, 20.0]); // [6/3, 60/3]
    }

    #[test]
    fn test_tensor_max_last_dim() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], &device);

        let b = client.max(&a, &[1], false).unwrap();

        assert_eq!(b.shape(), &[2]);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [5.0, 8.0]);
    }

    #[test]
    fn test_tensor_relu() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 1.0, -2.0], &[4], &device);

        let b = client.relu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_tensor_sigmoid() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);

        let b = client.sigmoid(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_silu() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device);
        let b = client.silu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // SiLU(x) = x / (1 + exp(-x))
        assert!((result[2] - 0.0).abs() < 1e-5); // SiLU(0) = 0
        assert!((result[3] - 0.7310586).abs() < 1e-4); // SiLU(1) ≈ 0.731
        assert!((result[1] - (-0.2689414)).abs() < 1e-4); // SiLU(-1) ≈ -0.269
    }

    #[test]
    fn test_tensor_gelu() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device);
        let b = client.gelu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // GELU(0) = 0
        assert!((result[2] - 0.0).abs() < 1e-5); // GELU(0) = 0
        assert!((result[3] - 0.8413).abs() < 0.01); // GELU(1) ≈ 0.841
        assert!((result[4] - 1.9545).abs() < 0.01); // GELU(2) ≈ 1.955
    }

    #[test]
    fn test_tensor_rms_norm() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Input: 2 rows, 4 features each
        let input = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            &[2, 4],
            &device,
        );
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);

        let out = client.rms_norm(&input, &weight, 1e-5).unwrap();
        let result: Vec<f32> = out.to_vec();

        // Row 1: [1, 2, 3, 4], RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.739
        // Normalized: [0.365, 0.730, 1.095, 1.460]
        let rms1 = (30.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[0] - 1.0 / rms1).abs() < 1e-4);
        assert!((result[1] - 2.0 / rms1).abs() < 1e-4);
        assert!((result[2] - 3.0 / rms1).abs() < 1e-4);
        assert!((result[3] - 4.0 / rms1).abs() < 1e-4);

        // Row 2: [2, 4, 6, 8], values are 2x row 1, so RMS is also 2x
        let rms2 = (120.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[4] - 2.0 / rms2).abs() < 1e-4);
    }

    #[test]
    fn test_tensor_layer_norm() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Input: 2 rows, 4 features each
        let input = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            &[2, 4],
            &device,
        );
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

        let out = client.layer_norm(&input, &weight, &bias, 1e-5).unwrap();
        let result: Vec<f32> = out.to_vec();

        // Row 1: [1, 2, 3, 4], mean = 2.5, var = 1.25, std = 1.118
        // Normalized: (x - mean) / std = [-1.342, -0.447, 0.447, 1.342]
        let mean1 = 2.5f32;
        let var1 = ((1.0 - mean1).powi(2)
            + (2.0 - mean1).powi(2)
            + (3.0 - mean1).powi(2)
            + (4.0 - mean1).powi(2))
            / 4.0;
        let std1 = (var1 + 1e-5).sqrt();
        assert!((result[0] - (1.0 - mean1) / std1).abs() < 1e-4);
        assert!((result[1] - (2.0 - mean1) / std1).abs() < 1e-4);
        assert!((result[2] - (3.0 - mean1) / std1).abs() < 1e-4);
        assert!((result[3] - (4.0 - mean1) / std1).abs() < 1e-4);

        // Verify normalized outputs sum to approximately 0 (zero-centered)
        let row1_sum: f32 = result[0..4].iter().sum();
        assert!(row1_sum.abs() < 1e-4);
    }

    #[test]
    fn test_tensor_argmax() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device);

        // argmax along dim=1 (find max index in each row)
        let out = client.argmax(&a, 1, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(result, [1, 2]); // Row 0: max at index 1 (5.0), Row 1: max at index 2 (6.0)

        // argmax along dim=0 (find max index in each column)
        let out = client.argmax(&a, 0, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(result, [1, 0, 1]); // Col 0: max at 1 (4.0), Col 1: max at 0 (5.0), Col 2: max at 1 (6.0)

        // Test keepdim=true
        let out = client.argmax(&a, 1, true).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(result, [1, 2]);
    }

    #[test]
    fn test_tensor_argmin() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device);

        // argmin along dim=1 (find min index in each row)
        let out = client.argmin(&a, 1, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(result, [0, 1]); // Row 0: min at index 0 (1.0), Row 1: min at index 1 (2.0)

        // argmin along dim=0 (find min index in each column)
        let out = client.argmin(&a, 0, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(result, [0, 1, 0]); // Col 0: min at 0 (1.0), Col 1: min at 1 (2.0), Col 2: min at 0 (3.0)

        // Test keepdim=true
        let out = client.argmin(&a, 1, true).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(result, [0, 1]);
    }

    #[test]
    fn test_tensor_softmax_last_dim() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        let b = client.softmax(&a, -1).unwrap();

        let result: Vec<f32> = b.to_vec();
        // Check that outputs sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Check monotonicity: result[0] < result[1] < result[2]
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_tensor_ops_i32() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[5i32, 6, 7, 8], &[4], &device);

        let c = client.add(&a, &b).unwrap();

        let result: Vec<i32> = c.to_vec();
        assert_eq!(result, [6, 8, 10, 12]);
    }

    #[test]
    fn test_tensor_dtype_mismatch() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device);

        let result = client.add(&a, &b);
        assert!(result.is_err());
    }

    // ===== New Unary Operations Tests =====

    #[test]
    fn test_tensor_tan() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Use 0.5 radians to avoid clippy::approx_constant warnings
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.5], &[2], &device);
        let b = client.tan(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert!((result[0] - 0.0).abs() < 1e-6); // tan(0) = 0
        // tan(0.5) ≈ 0.5463
        assert!((result[1] - 0.5463).abs() < 1e-3);
    }

    #[test]
    fn test_tensor_recip() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 4.0, 5.0], &[4], &device);
        let b = client.recip(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [1.0, 0.5, 0.25, 0.2]);
    }

    #[test]
    fn test_tensor_square() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, -4.0], &[4], &device);
        let b = client.square(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_tensor_floor() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.5f32, 2.9, -1.5, -2.9], &[4], &device);
        let b = client.floor(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [1.0, 2.0, -2.0, -3.0]);
    }

    #[test]
    fn test_tensor_ceil() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.5f32, 2.1, -1.5, -2.1], &[4], &device);
        let b = client.ceil(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [2.0, 3.0, -1.0, -2.0]);
    }

    #[test]
    fn test_tensor_round() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.4f32, 1.5, 2.5, -1.5], &[4], &device);
        let b = client.round(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // Rust's round() rounds half away from zero: 2.5 -> 3.0, -1.5 -> -2.0
        assert_eq!(result, [1.0, 2.0, 3.0, -2.0]);
    }

    // ===== New Element-wise Binary Operations Tests =====

    #[test]
    fn test_tensor_pow() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 0.5], &[3], &device);

        let c = client.pow(&a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [4.0, 9.0, 2.0]); // 2^2=4, 3^2=9, 4^0.5=2
    }

    #[test]
    fn test_tensor_maximum() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 8.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0, 7.0], &[4], &device);

        let c = client.maximum(&a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [2.0, 5.0, 6.0, 8.0]);
    }

    #[test]
    fn test_tensor_minimum() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 8.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0, 7.0], &[4], &device);

        let c = client.minimum(&a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [1.0, 4.0, 3.0, 7.0]);
    }

    // ===== ScalarOps Tests =====

    #[test]
    fn test_scalar_add() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = ScalarOps::add_scalar(&client, &a, 10.0).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_scalar_mul() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = ScalarOps::mul_scalar(&client, &a, 2.0).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scalar_pow() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = ScalarOps::pow_scalar(&client, &a, 2.0).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_scalar_div() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device);
        let b = ScalarOps::div_scalar(&client, &a, 10.0).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
    }

    // ===== CompareOps Tests =====

    #[test]
    fn test_compare_eq() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 3.0, 5.0], &[4], &device);

        let c = CompareOps::eq(&client, &a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [1.0, 0.0, 1.0, 0.0]); // 1=true, 0=false
    }

    #[test]
    fn test_compare_lt() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4], &device);

        let c = CompareOps::lt(&client, &a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [1.0, 0.0, 0.0, 0.0]); // 1<2, 2<2?, 3<2?, 4<2?
    }

    #[test]
    fn test_compare_gt() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4], &device);

        let c = CompareOps::gt(&client, &a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [0.0, 0.0, 1.0, 1.0]); // 1>2?, 2>2?, 3>2, 4>2
    }

    #[test]
    fn test_compare_le() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4], &device);

        let c = CompareOps::le(&client, &a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [1.0, 1.0, 0.0, 0.0]); // 1<=2, 2<=2, 3<=2?, 4<=2?
    }

    #[test]
    fn test_compare_ge() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4], &device);

        let c = CompareOps::ge(&client, &a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [0.0, 1.0, 1.0, 1.0]); // 1>=2?, 2>=2, 3>=2, 4>=2
    }

    #[test]
    fn test_compare_ne() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 3.0, 5.0], &[4], &device);

        let c = CompareOps::ne(&client, &a, &b).unwrap();

        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [0.0, 1.0, 0.0, 1.0]); // opposite of eq
    }

    #[test]
    fn test_compare_i32() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2i32, 2, 2, 2], &[4], &device);

        let c = CompareOps::lt(&client, &a, &b).unwrap();

        let result: Vec<i32> = c.to_vec();
        assert_eq!(result, [1, 0, 0, 0]);
    }

    // ===== Broadcasting Tests =====

    #[test]
    fn test_broadcast_scalar_to_vector() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [4] + [1] -> [4]
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[4]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_broadcast_vector_to_matrix_row() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [2, 3] + [3] -> [2, 3] (broadcast along rows)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0], &[3], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_vector_to_matrix_col() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [2, 3] + [2, 1] -> [2, 3] (broadcast along columns)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 100.0], &[2, 1], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [11.0, 12.0, 13.0, 104.0, 105.0, 106.0]);
    }

    #[test]
    fn test_broadcast_both_directions() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [3, 1] + [1, 4] -> [3, 4]
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[1, 4], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[3, 4]);
        let result: Vec<f32> = c.to_vec();
        // Row 0: 1 + [10, 20, 30, 40] = [11, 21, 31, 41]
        // Row 1: 2 + [10, 20, 30, 40] = [12, 22, 32, 42]
        // Row 2: 3 + [10, 20, 30, 40] = [13, 23, 33, 43]
        assert_eq!(
            result,
            [
                11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0
            ]
        );
    }

    #[test]
    fn test_broadcast_mul() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [2, 3] * [1] -> [2, 3]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

        let c = client.mul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_broadcast_sub() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [2, 2] - [2] -> [2, 2]
        let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);

        let c = client.sub(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [9.0, 18.0, 29.0, 38.0]);
    }

    #[test]
    fn test_broadcast_div() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [4] / [1] -> [4]
        let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

        let c = client.div(&a, &b).unwrap();

        assert_eq!(c.shape(), &[4]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_broadcast_3d() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [2, 2, 3] + [3] -> [2, 2, 3]
        let a = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 2, 3],
            &device,
        );
        let b = Tensor::<CpuRuntime>::from_slice(&[100.0f32, 200.0, 300.0], &[3], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2, 3]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(
            result,
            [
                101.0, 202.0, 303.0, 104.0, 205.0, 306.0, 107.0, 208.0, 309.0, 110.0, 211.0, 312.0
            ]
        );
    }

    #[test]
    fn test_broadcast_pow() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [3] ^ [1] -> [3]
        let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

        let c = client.pow(&a, &b).unwrap();

        assert_eq!(c.shape(), &[3]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_broadcast_maximum() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // max([2, 3], [3]) -> [2, 3]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 2.0, 4.0, 0.0, 6.0], &[2, 3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 3.0, 3.0], &[3], &device);

        let c = client.maximum(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [3.0, 5.0, 3.0, 4.0, 3.0, 6.0]);
    }

    #[test]
    fn test_broadcast_incompatible_shapes() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [3] + [4] -> Error (incompatible)
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);

        let result = client.add(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_i32() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [2, 2] + [2] -> [2, 2] (integer type)
        let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[10i32, 20], &[2], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<i32> = c.to_vec();
        assert_eq!(result, [11, 22, 13, 24]);
    }

    // ========================================================================
    // Comparison Broadcasting Tests
    // ========================================================================

    #[test]
    fn test_broadcast_compare_scalar() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Compare [4] with scalar [1] -> [4]
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.5f32], &[1], &device);

        // a > 2.5: [false, false, true, true] -> [0, 0, 1, 1]
        let c = client.gt(&a, &b).unwrap();
        assert_eq!(c.shape(), &[4]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [0.0, 0.0, 1.0, 1.0]);

        // a <= 2.5: [true, true, false, false] -> [1, 1, 0, 0]
        let c = client.le(&a, &b).unwrap();
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_broadcast_compare_matrix_row() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [2, 3] compared with [3] -> [2, 3]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device);

        // a < b: row0: [1<2, 2<3, 3<4]=[T,T,T], row1: [4<2, 5<3, 6<4]=[F,F,F]
        let c = client.lt(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_broadcast_compare_eq() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [3, 1] compared with [1, 3] -> [3, 3]
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);

        // Result is identity matrix (1s on diagonal)
        let c = client.eq(&a, &b).unwrap();
        assert_eq!(c.shape(), &[3, 3]);
        let result: Vec<f32> = c.to_vec();
        // Row 0: [1==1, 1==2, 1==3] = [1, 0, 0]
        // Row 1: [2==1, 2==2, 2==3] = [0, 1, 0]
        // Row 2: [3==1, 3==2, 3==3] = [0, 0, 1]
        assert_eq!(result, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    // ========================================================================
    // Half-Precision Tests (requires "f16" feature)
    // ========================================================================

    #[cfg(feature = "f16")]
    #[test]
    fn test_f16_tensor_add() {
        use half::f16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a_data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_data: Vec<f16> = vec![0.5, 1.5, 2.5, 3.5]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[4], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[4]);
        let result: Vec<f16> = c.to_vec();
        let expected: Vec<f16> = vec![1.5, 3.5, 5.5, 7.5]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(result, expected);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_f16_matmul() {
        use half::f16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a_data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_data: Vec<f16> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2, 2], &device);

        let c = TensorOps::matmul(&client, &a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f16> = c.to_vec();
        let expected: Vec<f16> = vec![19.0, 22.0, 43.0, 50.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(result, expected);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_bf16_tensor_mul() {
        use half::bf16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a_data: Vec<bf16> = vec![2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_data: Vec<bf16> = vec![1.5, 2.5, 3.5]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[3], &device);

        let c = client.mul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[3]);
        let result: Vec<bf16> = c.to_vec();
        let expected: Vec<bf16> = vec![3.0, 7.5, 14.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        assert_eq!(result, expected);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_f16_unary_ops() {
        use half::f16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let data: Vec<f16> = vec![1.0, 4.0, 9.0, 16.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let a = Tensor::<CpuRuntime>::from_slice(&data, &[4], &device);

        let sqrt_result = client.sqrt(&a).unwrap();
        let sqrt_vec: Vec<f16> = sqrt_result.to_vec();
        let expected_sqrt: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(sqrt_vec, expected_sqrt);

        let neg_result = client.neg(&a).unwrap();
        let neg_vec: Vec<f16> = neg_result.to_vec();
        let expected_neg: Vec<f16> = vec![-1.0, -4.0, -9.0, -16.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(neg_vec, expected_neg);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_f16_reduce() {
        use half::f16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device);

        let sum = client.sum(&a, &[1], false).unwrap();
        assert_eq!(sum.shape(), &[2]);
        let sum_vec: Vec<f16> = sum.to_vec();
        let expected: Vec<f16> = vec![6.0, 15.0].into_iter().map(f16::from_f32).collect();
        assert_eq!(sum_vec, expected);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_f16_broadcast() {
        use half::f16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a_data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_data: Vec<f16> = vec![10.0, 20.0].into_iter().map(f16::from_f32).collect();

        let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f16> = c.to_vec();
        let expected: Vec<f16> = vec![11.0, 22.0, 13.0, 24.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(result, expected);
    }

    // ===== FP8 Integration Tests =====

    #[cfg(feature = "fp8")]
    #[test]
    fn test_fp8e4m3_tensor_creation() {
        use crate::dtype::{DType, FP8E4M3};

        let device = CpuDevice::new();

        // Create FP8E4M3 tensor from slice
        let data: Vec<FP8E4M3> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(FP8E4M3::from_f32)
            .collect();

        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &device);

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), DType::FP8E4M3);
        assert_eq!(tensor.numel(), 4);

        // Verify data roundtrips
        let result: Vec<FP8E4M3> = tensor.to_vec();
        for (a, b) in data.iter().zip(result.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 0.1);
        }
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_fp8e5m2_tensor_creation() {
        use crate::dtype::{DType, FP8E5M2};

        let device = CpuDevice::new();

        // Create FP8E5M2 tensor from slice
        let data: Vec<FP8E5M2> = vec![10.0, 20.0, 30.0, 40.0]
            .into_iter()
            .map(FP8E5M2::from_f32)
            .collect();

        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[4], &device);

        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.dtype(), DType::FP8E5M2);

        // Verify data roundtrips (FP8E5M2 has less precision)
        let result: Vec<FP8E5M2> = tensor.to_vec();
        for (a, b) in data.iter().zip(result.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 5.0);
        }
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_fp8e4m3_add() {
        use crate::dtype::{DType, FP8E4M3};

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a_data: Vec<FP8E4M3> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(FP8E4M3::from_f32)
            .collect();
        let b_data: Vec<FP8E4M3> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(FP8E4M3::from_f32)
            .collect();

        let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2, 2], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.dtype(), DType::FP8E4M3);

        let result: Vec<FP8E4M3> = c.to_vec();
        let expected = [6.0, 8.0, 10.0, 12.0];
        for (val, exp) in result.iter().zip(expected.iter()) {
            // FP8E4M3 has ~20% tolerance
            assert!((val.to_f32() - exp).abs() < exp * 0.25 + 0.5);
        }
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_fp8e4m3_mul() {
        use crate::dtype::FP8E4M3;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a_data: Vec<FP8E4M3> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(FP8E4M3::from_f32)
            .collect();
        let b_data: Vec<FP8E4M3> = vec![2.0, 2.0, 2.0, 2.0]
            .into_iter()
            .map(FP8E4M3::from_f32)
            .collect();

        let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[4], &device);

        let c = client.mul(&a, &b).unwrap();

        let result: Vec<FP8E4M3> = c.to_vec();
        let expected = [2.0, 4.0, 6.0, 8.0];
        for (val, exp) in result.iter().zip(expected.iter()) {
            assert!((val.to_f32() - exp).abs() < exp * 0.25 + 0.5);
        }
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_fp8e5m2_large_values() {
        use crate::dtype::FP8E5M2;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // FP8E5M2 has larger dynamic range (~[-57344, 57344])
        let a_data: Vec<FP8E5M2> = vec![100.0, 200.0, 500.0, 1000.0]
            .into_iter()
            .map(FP8E5M2::from_f32)
            .collect();
        let b_data: Vec<FP8E5M2> = vec![2.0, 2.0, 2.0, 2.0]
            .into_iter()
            .map(FP8E5M2::from_f32)
            .collect();

        let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[4], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[4], &device);

        let c = client.mul(&a, &b).unwrap();

        let result: Vec<FP8E5M2> = c.to_vec();
        let expected = [200.0, 400.0, 1000.0, 2000.0];
        for (val, exp) in result.iter().zip(expected.iter()) {
            // FP8E5M2 has ~30% tolerance for larger values
            assert!((val.to_f32() - exp).abs() < exp * 0.35 + 10.0);
        }
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_fp8_full_scalar_tensor() {
        use crate::dtype::{DType, FP8E4M3};

        let device = CpuDevice::new();

        // Test Tensor::full_scalar with FP8E4M3
        let tensor = Tensor::<CpuRuntime>::full_scalar(&[2, 3], DType::FP8E4M3, 2.5, &device);

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::FP8E4M3);

        let result: Vec<FP8E4M3> = tensor.to_vec();
        for val in result {
            assert!((val.to_f32() - 2.5).abs() < 0.5);
        }
    }

    // ===== Cast Operation Tests =====

    #[test]
    fn test_cast_f32_to_f64() {
        use crate::dtype::DType;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.5f32, 2.5, 3.5, 4.5], &[2, 2], &device);
        let b = client.cast(&a, DType::F64).unwrap();

        assert_eq!(b.dtype(), DType::F64);
        assert_eq!(b.shape(), &[2, 2]);
        let result: Vec<f64> = b.to_vec();
        assert_eq!(result, [1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_cast_f64_to_i32() {
        use crate::dtype::DType;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.9f64, -2.1, 3.5, -4.9], &[4], &device);
        let b = client.cast(&a, DType::I32).unwrap();

        assert_eq!(b.dtype(), DType::I32);
        let result: Vec<i32> = b.to_vec();
        // Truncation toward zero
        assert_eq!(result, [1, -2, 3, -4]);
    }

    #[test]
    fn test_cast_i32_to_f32() {
        use crate::dtype::DType;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1i32, -2, 100, -50], &[4], &device);
        let b = client.cast(&a, DType::F32).unwrap();

        assert_eq!(b.dtype(), DType::F32);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [1.0, -2.0, 100.0, -50.0]);
    }

    #[test]
    fn test_cast_same_dtype_noop() {
        use crate::dtype::DType;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let b = client.cast(&a, DType::F32).unwrap();

        assert_eq!(b.dtype(), DType::F32);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_cast_f32_to_fp8e4m3() {
        use crate::dtype::{DType, FP8E4M3};

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 4.0, 8.0], &[4], &device);
        let b = client.cast(&a, DType::FP8E4M3).unwrap();

        assert_eq!(b.dtype(), DType::FP8E4M3);
        let result: Vec<FP8E4M3> = b.to_vec();

        // FP8 has limited precision, so check with tolerance
        assert!((result[0].to_f32() - 1.0).abs() < 0.1);
        assert!((result[1].to_f32() - 2.0).abs() < 0.2);
        assert!((result[2].to_f32() - 4.0).abs() < 0.5);
        assert!((result[3].to_f32() - 8.0).abs() < 1.0);
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_cast_fp8e4m3_to_f32() {
        use crate::dtype::{DType, FP8E4M3};

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Create FP8 tensor
        let fp8_data: Vec<FP8E4M3> = vec![
            FP8E4M3::from_f32(1.0),
            FP8E4M3::from_f32(2.0),
            FP8E4M3::from_f32(4.0),
            FP8E4M3::from_f32(8.0),
        ];
        let a = Tensor::<CpuRuntime>::from_slice(&fp8_data, &[4], &device);
        let b = client.cast(&a, DType::F32).unwrap();

        assert_eq!(b.dtype(), DType::F32);
        let result: Vec<f32> = b.to_vec();

        // Check roundtrip - values should be close
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.2);
        assert!((result[2] - 4.0).abs() < 0.5);
        assert!((result[3] - 8.0).abs() < 1.0);
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_cast_f32_to_fp8e5m2() {
        use crate::dtype::{DType, FP8E5M2};

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // E5M2 has larger range but less precision
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 100.0, 1000.0, 10000.0], &[4], &device);
        let b = client.cast(&a, DType::FP8E5M2).unwrap();

        assert_eq!(b.dtype(), DType::FP8E5M2);
        let result: Vec<FP8E5M2> = b.to_vec();

        // E5M2 can represent larger values but with less precision
        assert!((result[0].to_f32() - 1.0).abs() < 0.5);
        assert!((result[1].to_f32() - 100.0).abs() < 50.0);
        assert!((result[2].to_f32() - 1000.0).abs() < 500.0);
        assert!((result[3].to_f32() - 10000.0).abs() < 5000.0);
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_cast_fp8e4m3_to_fp8e5m2() {
        use crate::dtype::{DType, FP8E4M3, FP8E5M2};

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Create FP8E4M3 tensor
        let fp8_data: Vec<FP8E4M3> = vec![
            FP8E4M3::from_f32(1.0),
            FP8E4M3::from_f32(2.0),
            FP8E4M3::from_f32(4.0),
        ];
        let a = Tensor::<CpuRuntime>::from_slice(&fp8_data, &[3], &device);

        // Cast E4M3 -> E5M2
        let b = client.cast(&a, DType::FP8E5M2).unwrap();
        assert_eq!(b.dtype(), DType::FP8E5M2);

        let result: Vec<FP8E5M2> = b.to_vec();
        // Values should be preserved (approximately)
        assert!((result[0].to_f32() - 1.0).abs() < 0.5);
        assert!((result[1].to_f32() - 2.0).abs() < 1.0);
        assert!((result[2].to_f32() - 4.0).abs() < 2.0);
    }
}
