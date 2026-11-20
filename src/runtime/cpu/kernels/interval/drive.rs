//! Chunked lane loops that accumulate one status flag.

use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// One lane input: an element address and a step of zero or one element.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    addr: usize,
    step: usize,
}

impl Input {
    /// Input starting at `addr` that advances `step` elements per lane.
    pub fn new(addr: u64, step: u32) -> Self {
        Self {
            addr: addr as usize,
            step: step as usize,
        }
    }

    /// Read lane `index`.
    ///
    /// # Safety
    ///
    /// The input addresses `index * step + 1` readable elements of `T`.
    #[inline(always)]
    pub unsafe fn get<T: Copy>(self, index: usize) -> T {
        unsafe { *(self.addr as *const T).add(index * self.step) }
    }

    /// The start address when every lane advances one element.
    pub fn dense(self) -> Option<usize> {
        (self.step == 1).then_some(self.addr)
    }
}

/// Read lane `index` of a dense input address.
///
/// # Safety
///
/// The address holds more than `index` readable elements of `T`.
#[inline(always)]
pub unsafe fn dense_get<T: Copy>(addr: usize, index: usize) -> T {
    unsafe { *(addr as *const T).add(index) }
}

/// One contiguous lane output.
#[derive(Debug, Clone, Copy)]
pub struct Output(usize);

impl Output {
    /// Output for a fresh contiguous tensor.
    pub fn of(tensor: &Tensor<CpuRuntime>) -> Self {
        Self(tensor.ptr() as usize)
    }

    /// Write lane `index`.
    ///
    /// # Safety
    ///
    /// The output holds more than `index` writable elements of `T`, and no other lane writes `index`.
    #[inline(always)]
    pub unsafe fn put<T: Copy>(self, index: usize, value: T) {
        unsafe { *(self.0 as *mut T).add(index) = value }
    }
}

/// Run `lane` for every index below `len` and return whether any lane flagged.
///
/// Chunks run in parallel when the client's gate allows it.
pub fn run_lanes<F>(client: &CpuClient, len: usize, lane: F) -> bool
where
    F: Fn(usize) -> bool + Send + Sync,
{
    if len == 0 {
        return false;
    }
    let chunk = client.parallelism().chunk_size.unwrap_or(4096).max(1);
    if len <= chunk {
        return run_range(&lane, 0, len);
    }
    let flagged = AtomicBool::new(false);
    client.par_for_each(len.div_ceil(chunk), len, |block| {
        let start = block * chunk;
        let end = (start + chunk).min(len);
        if run_range(&lane, start, end) {
            flagged.store(true, Ordering::Relaxed);
        }
    });
    flagged.into_inner()
}

/// Run every lane in `start..end`, without short-circuiting, and report any flag.
#[inline(always)]
fn run_range<F: Fn(usize) -> bool>(lane: &F, start: usize, end: usize) -> bool {
    let mut flagged = false;
    for index in start..end {
        flagged |= lane(index);
    }
    flagged
}

/// Set a validated one-element U32 status flag when `flagged` holds.
pub fn raise(status: &Tensor<CpuRuntime>, flagged: bool) {
    if flagged {
        // Validation guarantees one U32 element at this address.
        unsafe { *(status.ptr() as *mut u32) = 1 }
    }
}
