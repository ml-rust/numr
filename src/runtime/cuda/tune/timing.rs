//! Time repeated launches on the compute stream with CUDA events.

use cudarc::driver::CudaEvent;
use cudarc::driver::sys::CUevent_flags;

use super::super::client::CudaClient;
use crate::error::{Error, Result};

/// Time `launch` on the client's compute stream and return the fastest
/// iteration in microseconds.
///
/// Runs `launch` once as a warm-up, then `iters` timed iterations. Each
/// iteration is bracketed by its own pair of timing-enabled events, and
/// the result is the minimum over iterations: the estimator that best
/// tracks kernel cost under a loaded machine, since interference only ever
/// adds time. The stream is synchronized once before the events are read.
///
/// `iters` must be at least 1.
pub fn time_launches(
    client: &CudaClient,
    iters: usize,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<f32> {
    if iters == 0 {
        return Err(Error::InvalidArgument {
            arg: "iters",
            reason: "time_launches needs at least one iteration".into(),
        });
    }

    let context = client.context();
    let stream = client.stream();

    launch()?;
    stream.synchronize()?;

    let mut pairs: Vec<(CudaEvent, CudaEvent)> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(stream)?;
        launch()?;
        end.record(stream)?;
        pairs.push((start, end));
    }
    stream.synchronize()?;

    let mut best = f32::INFINITY;
    for (start, end) in &pairs {
        let micros = start.elapsed_ms(end)? * 1000.0;
        if micros < best {
            best = micros;
        }
    }
    Ok(best)
}
