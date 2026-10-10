//! Make a client's CUDA context current on the calling thread.
//!
//! The driver keeps one current context per thread, and a thread starts with
//! none. A driver call that takes no stream or event, such as an event
//! create or a context synchronize, resolves its context from that slot and
//! fails with `CUDA_ERROR_INVALID_CONTEXT` on a thread that has none. On a
//! thread where another device's context is current, it acts on that device.
//!
//! cudarc's safe calls bind their context themselves. Every numr entry point
//! that calls the driver API directly binds through [`bind_context`] first.

use cudarc::driver::safe::CudaContext;
use cudarc::driver::{DriverError, result};

/// Make `ctx` the calling thread's current context.
///
/// Costs one context query when `ctx` is already current.
///
/// Unlike `CudaContext::bind_to_thread`, this leaves cudarc's recorded
/// asynchronous error in place, so cudarc's next safe call still reports it.
///
/// # Errors
///
/// Returns the driver's error when the current context cannot be read or
/// set, for example after the driver shuts down.
#[inline]
pub(crate) fn bind_context(ctx: &CudaContext) -> Result<(), DriverError> {
    let wanted = ctx.cu_ctx();
    if result::ctx::get_current()? != Some(wanted) {
        // SAFETY: `wanted` belongs to a live `CudaContext`, which keeps the
        // driver context retained for as long as `ctx` is borrowed.
        unsafe { result::ctx::set_current(wanted) }?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cuda::{CudaClient, CudaDevice};

    /// A thread that has made no CUDA call has no current context. After
    /// the bind, the client's context is current there.
    #[test]
    fn binds_on_a_fresh_thread() {
        let Ok(client) = CudaClient::new(CudaDevice::new(0)) else {
            return;
        };
        let ctx = client.context().clone();
        // Raw context handles are not `Send`; they cross the join as addresses.
        let worker = std::thread::spawn(move || {
            let before = result::ctx::get_current().expect("query before");
            bind_context(&ctx).expect("bind");
            let after = result::ctx::get_current().expect("query after");
            bind_context(&ctx).expect("bind again");
            (
                before.map(|c| c as usize),
                after.map(|c| c as usize),
                ctx.cu_ctx() as usize,
            )
        });
        let (before, after, wanted) = worker.join().expect("worker thread");
        assert_eq!(before, None, "a fresh thread already had a current context");
        assert_eq!(after, Some(wanted), "the bind did not take effect");
    }
}
