//! The output layout that the first forward of a multi-output segment records.
//!
//! Each recompute must return the same layout, or backward fails with an error.

use std::sync::{Arc, Mutex};

use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;

/// Shape and dtype of one segment output.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Slot {
    pub(super) shape: Vec<usize>,
    pub(super) dtype: DType,
}

impl Slot {
    /// Element count of the output.
    pub(super) fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The output slots the first forward records. Each recompute is checked against them.
pub(super) type Record = Arc<Mutex<Option<Vec<Slot>>>>;

/// Read the shape and dtype of every output, and refuse non-float outputs.
pub(super) fn describe<R: Runtime<DType = DType>>(outputs: &[Var<R>]) -> Result<Vec<Slot>> {
    if outputs.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "f",
            reason: "checkpoint_many_with_client: the segment returned no outputs. \
                     Return at least one float Var from the closure."
                .to_string(),
        });
    }
    outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            let dtype = output.tensor().dtype();
            if !dtype.is_float() {
                return Err(Error::InvalidArgument {
                    arg: "f",
                    reason: format!(
                        "checkpoint_many_with_client: output {index} has dtype {dtype:?}, \
                         and only float outputs can be checkpointed. Compute this output \
                         outside the segment."
                    ),
                });
            }
            Ok(Slot {
                shape: output.shape().to_vec(),
                dtype,
            })
        })
        .collect()
}

/// Store the slots of the first forward, or check a recompute against them.
pub(super) fn remember(record: &Record, slots: &[Slot]) -> Result<()> {
    let mut guard = record.lock().map_err(|_| poisoned())?;
    match guard.as_deref() {
        None => {
            *guard = Some(slots.to_vec());
            Ok(())
        }
        Some(first) if first == slots => Ok(()),
        Some(first) => Err(Error::InvalidArgument {
            arg: "f",
            reason: format!(
                "checkpoint_many_with_client: the first forward returned {} but the \
                 recompute returned {}. The segment must return the same shapes and \
                 dtypes on every call. Remove the state or randomness that changes them.",
                list(first),
                list(slots)
            ),
        }),
    }
}

/// The slots the first forward recorded.
pub(super) fn recorded(record: &Record) -> Result<Vec<Slot>> {
    record
        .lock()
        .map_err(|_| poisoned())?
        .clone()
        .ok_or_else(|| {
            Error::Internal(
                "checkpoint_many_with_client: the forward returned without recording its outputs"
                    .to_string(),
            )
        })
}

/// The error for a poisoned output-record lock.
fn poisoned() -> Error {
    Error::Internal(
        "checkpoint_many_with_client: the output record lock is poisoned, \
         because an earlier segment call panicked"
            .to_string(),
    )
}

/// Format slots as `[F32 [2, 3], BF16 []]` for an error message.
fn list(slots: &[Slot]) -> String {
    let parts: Vec<String> = slots
        .iter()
        .map(|slot| format!("{:?} {:?}", slot.dtype, slot.shape))
        .collect();
    format!("[{}]", parts.join(", "))
}
