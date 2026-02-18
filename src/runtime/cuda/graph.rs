//! CUDA graph capture and replay
//!
//! Wraps cudarc's `CudaGraph` with `Send + Sync + Clone` for use with
//! numr's `Graph` trait.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::safe::CudaGraph as CudarcGraph;

/// Wrapper to make cudarc's CudaGraph safe to send across threads.
///
/// # Safety
///
/// cudarc's `CudaGraph` contains raw CUDA pointers (`CUgraph`, `CUgraphExec`)
/// which don't auto-implement `Send`. We wrap it in `Mutex` to serialize all
/// access. The only operation after instantiation is `launch()`, which:
/// 1. Binds the CUDA context to the current thread (`ctx.bind_to_thread()`)
/// 2. Calls `cuGraphLaunch` (a stream-ordered operation)
///
/// No concurrent graph structure modification ever occurs.
struct CudaGraphInner(CudarcGraph);

// SAFETY: Access is serialized via Mutex. After instantiation, only launch()
// is called, which binds CUDA context to the calling thread.
unsafe impl Send for CudaGraphInner {}

/// CUDA graph — a captured computation sequence replayed via `cuGraphLaunch`.
///
/// Created by `CudaRuntime::capture_graph()`. Thread-safe via internal `Mutex`.
/// `Clone` bumps the `Arc` refcount (no graph duplication).
pub struct CudaGraph {
    inner: Arc<Mutex<CudaGraphInner>>,
    launch_count: Arc<AtomicUsize>,
}

impl Clone for CudaGraph {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            launch_count: self.launch_count.clone(),
        }
    }
}

impl std::fmt::Debug for CudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraph")
            .field("launch_count", &self.launch_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl CudaGraph {
    /// Create a new CudaGraph wrapping cudarc's graph.
    pub(crate) fn new(graph: CudarcGraph) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CudaGraphInner(graph))),
            launch_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// How many times this graph has been launched.
    pub fn launch_count(&self) -> usize {
        self.launch_count.load(Ordering::Relaxed)
    }
}

impl crate::runtime::Graph for CudaGraph {
    fn launch(&self) -> crate::error::Result<()> {
        let guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        guard
            .0
            .launch()
            .map_err(|e| crate::error::Error::Backend(format!("CUDA graph launch failed: {e}")))?;
        self.launch_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn is_replay_capable(&self) -> bool {
        true
    }
}

// SAFETY: All interior access is serialized via Mutex. Arc provides shared ownership.
// The CudaGraph is only ever launched (no structural modification after instantiation).
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}
