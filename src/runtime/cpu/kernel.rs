//! Kernel trait implementation for CPU runtime

use super::CpuRuntime;
use super::client::CpuClient;
use super::kernels;
use crate::dtype::Element;
use crate::ops::{BinaryOp, Kernel, ReduceOp, UnaryOp};

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
