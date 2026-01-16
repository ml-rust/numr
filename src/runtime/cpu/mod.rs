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

mod kernels;

use super::{DefaultAllocator, Device, Runtime, RuntimeClient};
use crate::dtype::Element;
use crate::ops::{BinaryOp, Kernel, ReduceOp, UnaryOp};
use std::alloc::{Layout as AllocLayout, alloc_zeroed, dealloc};

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
    device: CpuDevice,
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
// TensorOps Implementation
// ============================================================================

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    CompareOps, ScalarOps, TensorOps, broadcast_shape, matmul_output_shape, normalize_softmax_dim,
    reduce_output_shape,
};
use crate::tensor::Tensor;

/// Macro for dtype dispatch to typed kernel calls
///
/// This macro matches on dtype and executes the code block with the appropriate type.
/// Usage: dispatch_dtype!(dtype, T => { code using T }, "op_name")
macro_rules! dispatch_dtype {
    ($dtype:expr, $T:ident => $body:block, $error_op:expr) => {
        match $dtype {
            DType::F64 => {
                type $T = f64;
                $body
            }
            DType::F32 => {
                type $T = f32;
                $body
            }
            DType::I64 => {
                type $T = i64;
                $body
            }
            DType::I32 => {
                type $T = i32;
                $body
            }
            DType::I16 => {
                type $T = i16;
                $body
            }
            DType::I8 => {
                type $T = i8;
                $body
            }
            DType::U64 => {
                type $T = u64;
                $body
            }
            DType::U32 => {
                type $T = u32;
                $body
            }
            DType::U16 => {
                type $T = u16;
                $body
            }
            DType::U8 => {
                type $T = u8;
                $body
            }
            DType::F16 | DType::BF16 | DType::Bool => {
                return Err(Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $error_op,
                })
            }
        }
    };
}

/// TensorOps implementation for CPU runtime.
///
/// # Note on Broadcasting
///
/// Binary operations currently require tensors to have identical shapes.
/// Broadcasting support is planned for a future release.
impl TensorOps<CpuRuntime> for CpuClient {
    // ===== Binary Operations =====

    fn add(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Add, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Sub, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Mul, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Div, a, b, "div")
    }

    // ===== Unary Operations =====

    fn neg(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Neg, a, "neg")
    }

    fn abs(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Abs, a, "abs")
    }

    fn sqrt(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sqrt, a, "sqrt")
    }

    fn exp(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Exp, a, "exp")
    }

    fn log(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Log, a, "log")
    }

    fn sin(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sin, a, "sin")
    }

    fn cos(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Cos, a, "cos")
    }

    fn tanh(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Tanh, a, "tanh")
    }

    fn tan(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Tan, a, "tan")
    }

    fn recip(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Recip, a, "recip")
    }

    fn square(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Square, a, "square")
    }

    fn floor(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Floor, a, "floor")
    }

    fn ceil(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Ceil, a, "ceil")
    }

    fn round(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Round, a, "round")
    }

    // ===== Element-wise Binary (extended) =====

    fn pow(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Pow, a, b, "pow")
    }

    fn maximum(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Max, a, b, "maximum")
    }

    fn minimum(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Min, a, b, "minimum")
    }

    // ===== Matrix Operations =====

    fn matmul(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        let dtype = a.dtype();

        // Compute output shape
        let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        })?;

        // Get matrix dimensions (last two dims)
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // For now, require contiguous tensors
        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);

        // Calculate batch size
        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        // Create output tensor
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &self.device);

        let a_ptr = a_contig.storage().ptr();
        let b_ptr = b_contig.storage().ptr();
        let out_ptr = out.storage().ptr();

        // Leading dimensions for contiguous row-major matrices
        let lda = k;
        let ldb = n;
        let ldc = n;

        // Dispatch based on dtype
        dispatch_dtype!(dtype, T => {
            unsafe {
                for batch in 0..batch_size {
                    let a_offset = batch * m * k;
                    let b_offset = batch * k * n;
                    let out_offset = batch * m * n;

                    <Self as Kernel<CpuRuntime>>::matmul::<T>(
                        self,
                        (a_ptr as *const T).add(a_offset),
                        (b_ptr as *const T).add(b_offset),
                        (out_ptr as *mut T).add(out_offset),
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    );
                }
            }
        }, "matmul");

        Ok(out)
    }

    // ===== Reductions =====

    fn sum(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Sum, a, dims, keepdim, "sum")
    }

    fn mean(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Mean, a, dims, keepdim, "mean")
    }

    fn max(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Max, a, dims, keepdim, "max")
    }

    fn min(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Min, a, dims, keepdim, "min")
    }

    // ===== Activations =====

    fn relu(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::ReLU, "relu")
    }

    fn sigmoid(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Sigmoid, "sigmoid")
    }

    fn softmax(&self, a: &Tensor<CpuRuntime>, dim: isize) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let ndim = a.ndim();

        // Normalize dimension
        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &self.device);

        let shape = a.shape();

        // Calculate outer_size (product of dims before softmax dim)
        // and dim_size (size of softmax dim)
        // and inner_size (product of dims after softmax dim)
        let outer_size: usize = shape[..dim_idx].iter().product();
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product();

        // For softmax, we need the data laid out so that the softmax dimension is contiguous
        // If dim is the last dimension, we can use the simple kernel
        // Otherwise, we need to iterate

        if dim_idx == ndim - 1 {
            // Simple case: softmax over last dimension
            let a_ptr = a_contig.storage().ptr();
            let out_ptr = out.storage().ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    kernels::softmax_kernel::<T>(
                        a_ptr as *const T,
                        out_ptr as *mut T,
                        outer_size,
                        dim_size,
                    );
                }
            }, "softmax");
        } else {
            // General case: softmax over non-last dimension
            // Pre-allocate buffer outside loops to avoid repeated allocations
            let a_ptr = a_contig.storage().ptr();
            let out_ptr = out.storage().ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    // Pre-allocate reusable buffer for softmax computation
                    let mut slice = vec![0.0f64; dim_size];

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            // Elements are at: outer * dim_size * inner_size + d * inner_size + inner
                            let base_idx = outer * dim_size * inner_size + inner;
                            let stride = inner_size;

                            // Read slice into buffer
                            for (d, slot) in slice.iter_mut().enumerate() {
                                let idx = base_idx + d * stride;
                                *slot = (*(a_ptr as *const T).add(idx)).to_f64();
                            }

                            // Compute softmax with numerical stability
                            let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let mut exp_sum = 0.0f64;
                            for val in &mut slice {
                                *val = (*val - max_val).exp();
                                exp_sum += *val;
                            }

                            // Handle edge case: avoid division by zero
                            let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };

                            // Write normalized values back
                            for (d, &val) in slice.iter().enumerate() {
                                let idx = base_idx + d * stride;
                                *(out_ptr as *mut T).add(idx) = T::from_f64(val * inv_sum);
                            }
                        }
                    }
                }
            }, "softmax");
        }

        Ok(out)
    }
}

// ============================================================================
// ScalarOps Implementation
// ============================================================================

impl ScalarOps<CpuRuntime> for CpuClient {
    fn add_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Add, a, scalar, "add_scalar")
    }

    fn sub_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Sub, a, scalar, "sub_scalar")
    }

    fn mul_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Mul, a, scalar, "mul_scalar")
    }

    fn div_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Div, a, scalar, "div_scalar")
    }

    fn pow_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Pow, a, scalar, "pow_scalar")
    }
}

// ============================================================================
// CompareOps Implementation
// ============================================================================

impl CompareOps<CpuRuntime> for CpuClient {
    fn eq(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, kernels::CompareOp::Eq, a, b, "eq")
    }

    fn ne(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, kernels::CompareOp::Ne, a, b, "ne")
    }

    fn lt(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, kernels::CompareOp::Lt, a, b, "lt")
    }

    fn le(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, kernels::CompareOp::Le, a, b, "le")
    }

    fn gt(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, kernels::CompareOp::Gt, a, b, "gt")
    }

    fn ge(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, kernels::CompareOp::Ge, a, b, "ge")
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Ensure a tensor is contiguous, cloning if already contiguous or copying if not.
///
/// This is a common pattern used before kernel dispatch to ensure memory layout
/// is suitable for efficient computation.
#[inline]
fn ensure_contiguous(tensor: &Tensor<CpuRuntime>) -> Tensor<CpuRuntime> {
    if tensor.is_contiguous() {
        tensor.clone()
    } else {
        tensor.contiguous()
    }
}

/// Validate that two tensors have matching dtypes for binary operations.
#[inline]
fn validate_binary_dtypes(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<DType> {
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    Ok(a.dtype())
}

/// Compute broadcast shape for binary operations.
///
/// Returns the output shape after broadcasting, or an error if shapes are incompatible.
#[inline]
fn compute_broadcast_shape(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Vec<usize>> {
    broadcast_shape(a.shape(), b.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })
}

fn binary_op_impl(
    client: &CpuClient,
    op: BinaryOp,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // Create output tensor
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
    let out_ptr = out.storage().ptr();

    // Check if we can use the fast path (same shapes, both contiguous)
    let same_shapes = a.shape() == b.shape() && a.shape() == out_shape.as_slice();
    let both_contiguous = a.is_contiguous() && b.is_contiguous();

    if same_shapes && both_contiguous {
        // Fast path: no broadcasting needed, use contiguous kernel
        let len = a.numel();
        let a_ptr = a.storage().ptr();
        let b_ptr = b.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::binary_op::<T>(
                    client, op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                );
            }
        }, op_name);
    } else {
        // Broadcasting path: use strided kernel
        // Broadcast both inputs to output shape (zero-copy views with stride 0 for broadcast dims)
        let a_broadcast = a.broadcast_to(&out_shape)?;
        let b_broadcast = b.broadcast_to(&out_shape)?;

        let a_ptr = a_broadcast.storage().ptr();
        let b_ptr = b_broadcast.storage().ptr();

        // Get strides from broadcast layouts
        let a_strides: Vec<isize> = a_broadcast.layout().strides().to_vec();
        let b_strides: Vec<isize> = b_broadcast.layout().strides().to_vec();
        let a_offset = a_broadcast.layout().offset();
        let b_offset = b_broadcast.layout().offset();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::binary_op_strided_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    &out_shape,
                    &a_strides,
                    &b_strides,
                    a_offset,
                    b_offset,
                );
            }
        }, op_name);
    }

    Ok(out)
}

fn unary_op_impl(
    client: &CpuClient,
    op: UnaryOp,
    a: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device);

    let len = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            <CpuClient as Kernel<CpuRuntime>>::unary_op::<T>(
                client, op,
                a_ptr as *const T,
                out_ptr as *mut T,
                len,
            );
        }
    }, op_name);

    Ok(out)
}

fn scalar_op_impl(
    client: &CpuClient,
    op: BinaryOp,
    a: &Tensor<CpuRuntime>,
    scalar: f64,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device);

    let len = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::scalar_op_kernel::<T>(
                op,
                a_ptr as *const T,
                scalar,
                out_ptr as *mut T,
                len,
            );
        }
    }, op_name);

    Ok(out)
}

fn compare_op_impl(
    client: &CpuClient,
    op: kernels::CompareOp,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
    let out_ptr = out.storage().ptr();

    // Fast path for same shapes, both contiguous
    let same_shapes = a.shape() == b.shape() && a.shape() == out_shape.as_slice();
    let both_contiguous = a.is_contiguous() && b.is_contiguous();

    if same_shapes && both_contiguous {
        let len = a.numel();
        let a_ptr = a.storage().ptr();
        let b_ptr = b.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::compare_op_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                );
            }
        }, op_name);
    } else {
        // Broadcasting path: use strided kernel
        let a_broadcast = a.broadcast_to(&out_shape)?;
        let b_broadcast = b.broadcast_to(&out_shape)?;

        let a_strides: Vec<isize> = a_broadcast.layout().strides().to_vec();
        let b_strides: Vec<isize> = b_broadcast.layout().strides().to_vec();
        let a_offset = a_broadcast.layout().offset();
        let b_offset = b_broadcast.layout().offset();
        let a_ptr = a_broadcast.storage().ptr();
        let b_ptr = b_broadcast.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::compare_op_strided_kernel::<T>(
                    op,
                    a_ptr as *const T,
                    b_ptr as *const T,
                    out_ptr as *mut T,
                    &out_shape,
                    &a_strides,
                    &b_strides,
                    a_offset,
                    b_offset,
                );
            }
        }, op_name);
    }

    Ok(out)
}

/// Activation operation kind for kernel dispatch
#[derive(Copy, Clone)]
enum ActivationOp {
    ReLU,
    Sigmoid,
}

/// Helper for activation operations (relu, sigmoid)
fn activation_op_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    op: ActivationOp,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device);

    let len = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            match op {
                ActivationOp::ReLU => kernels::relu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::Sigmoid => kernels::sigmoid_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
            }
        }
    }, op_name);

    Ok(out)
}

fn reduce_impl(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimensions
    for &d in dims {
        if d >= ndim {
            return Err(Error::InvalidDimension {
                dim: d as isize,
                ndim,
            });
        }
    }

    // For now, only support reducing the last dimension when contiguous
    // Full reduction support requires permuting data or strided kernels
    if dims.len() == 1 && dims[0] == ndim - 1 && a.is_contiguous() {
        // Simple case: reduce last dimension
        let reduce_size = shape[ndim - 1];
        let outer_size: usize = shape[..ndim - 1].iter().product();
        let outer_size = outer_size.max(1);

        let out_shape = reduce_output_shape(shape, dims, keepdim);
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

        let a_ptr = a.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::reduce::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, op_name);

        Ok(out)
    } else if dims.is_empty() {
        // No dimensions to reduce - return a copy
        Ok(a.clone())
    } else {
        // General case: need to handle arbitrary dimensions
        // For now, make contiguous and reduce sequentially
        let a_contig = ensure_contiguous(a);

        // Reduce one dimension at a time, from highest to lowest
        let mut sorted_dims: Vec<usize> = dims.to_vec();
        sorted_dims.sort_unstable();
        sorted_dims.reverse();

        let mut current = a_contig;
        for &dim in &sorted_dims {
            current = reduce_single_dim(client, op, &current, dim, keepdim, op_name)?;
        }

        Ok(current)
    }
}

/// Reduce a single dimension
fn reduce_single_dim(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    keepdim: bool,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = shape[dim + 1..].iter().product();
    let inner_size = inner_size.max(1);

    let out_shape = reduce_output_shape(shape, &[dim], keepdim);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    // If reducing non-last dimension, we need special handling
    if dim == ndim - 1 {
        // Reducing last dimension - can use kernel directly
        let a_ptr = a.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::reduce::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, op_name);
    } else {
        // Reducing non-last dimension - iterate manually
        let a_ptr = a.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                reduce_non_last_dim::<T>(
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    outer_size,
                    reduce_size,
                    inner_size,
                );
            }
        }, op_name);
    }

    Ok(out)
}

/// Reduce a non-last dimension
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn reduce_non_last_dim<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => T::zero(),
                ReduceOp::Prod => T::one(),
                ReduceOp::Max => {
                    let idx = outer * reduce_size * inner_size + inner;
                    *a.add(idx)
                }
                ReduceOp::Min => {
                    let idx = outer * reduce_size * inner_size + inner;
                    *a.add(idx)
                }
                ReduceOp::All => T::one(),
                ReduceOp::Any => T::zero(),
            };

            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = *a.add(idx);

                acc = match op {
                    ReduceOp::Sum | ReduceOp::Mean => acc + val,
                    ReduceOp::Prod => acc * val,
                    ReduceOp::Max => {
                        if val > acc {
                            val
                        } else {
                            acc
                        }
                    }
                    ReduceOp::Min => {
                        if val < acc {
                            val
                        } else {
                            acc
                        }
                    }
                    ReduceOp::All => {
                        if val.to_f64() != 0.0 && acc.to_f64() != 0.0 {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }
                    ReduceOp::Any => {
                        if val.to_f64() != 0.0 || acc.to_f64() != 0.0 {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }
                };
            }

            // Apply mean scaling if needed
            if matches!(op, ReduceOp::Mean) {
                acc = T::from_f64(acc.to_f64() / reduce_size as f64);
            }

            let out_idx = outer * inner_size + inner;
            *out.add(out_idx) = acc;
        }
    }
}

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
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &device,
        );
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
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &device,
        );
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
            [11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0]
        );
    }

    #[test]
    fn test_broadcast_mul() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // [2, 3] * [1] -> [2, 3]
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &device,
        );
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
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
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
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 5.0, 2.0, 4.0, 0.0, 6.0],
            &[2, 3],
            &device,
        );
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
}
