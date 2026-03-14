//! Graph capture and replay for compute backends
//!
//! Graph capture records a sequence of operations that can be replayed efficiently.
//! This is a runtime-level concept (CUDA Graphs, Vulkan command buffers, etc.)
//! that benefits any compute workload — not just ML.

/// A captured computation sequence that can be replayed.
///
/// # Replay semantics
///
/// On capture-capable backends (CUDA), `launch()` replays the recorded
/// computation on the same fixed-address buffers. Callers update input
/// data in-place, then call `launch()` to re-execute with new values.
///
/// On non-capture backends (CPU, WebGPU), `capture_graph` executes the
/// closure eagerly and returns `NoOpGraph`. `launch()` is a no-op —
/// the computation already ran. Callers wanting repeated execution on
/// these backends must call the operations directly (not via launch).
///
/// Use `R::supports_graph_capture()` to check capability without
/// side effects, then branch:
///
/// ```ignore
/// if R::supports_graph_capture() {
///     let (graph, _) = R::capture_graph(client, |c| hot_path(c))?;
///     loop { update_inputs(); graph.launch()?; read_outputs(); }
/// } else {
///     loop { update_inputs(); hot_path(client)?; }
/// }
/// ```
pub trait Graph: Send + Sync + Clone {
    /// Replay the recorded computation.
    fn launch(&self) -> crate::error::Result<()>;

    /// Whether `launch()` actually replays computation.
    ///
    /// Returns `true` for backends with real capture (CUDA), `false` for no-op (CPU, WebGPU).
    ///
    /// # Invariant
    ///
    /// Must be consistent with `Runtime::supports_graph_capture()`:
    /// if `supports_graph_capture()` returns true, then any `Graph` produced
    /// by `capture_graph()` MUST return true from `is_replay_capable()`,
    /// and vice versa.
    fn is_replay_capable(&self) -> bool {
        false
    }
}

/// No-op graph for backends without capture support (CPU, WebGPU).
///
/// Operations execute eagerly during "capture" — `launch()` is a no-op.
#[derive(Clone, Debug, Default)]
pub struct NoOpGraph;

impl Graph for NoOpGraph {
    fn launch(&self) -> crate::error::Result<()> {
        Ok(())
    }
    // is_replay_capable() returns false (default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_graph_launch() {
        let graph = NoOpGraph;
        assert!(graph.launch().is_ok());
        assert!(!graph.is_replay_capable());
    }

    #[test]
    fn test_noop_graph_clone() {
        let graph = NoOpGraph;
        let cloned = graph.clone();
        assert!(cloned.launch().is_ok());
    }

    #[test]
    fn test_noop_graph_send_sync() {
        fn assert_send_sync<T: Send + Sync + Clone>() {}
        assert_send_sync::<NoOpGraph>();
    }
}
