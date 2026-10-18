//! One round of a collective: what each rank posts, how the last arrival checks and computes
//! the round, and how each rank takes its own result.

use super::{host, reduce};
use crate::dtype::DType;
use crate::error::Error;
use crate::runtime::communicator::ReduceOp;

/// The collective a rank asked for, with every parameter that must match across ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    AllReduce(ReduceOp),
    Broadcast { root: usize },
    AllGather,
    ReduceScatter(ReduceOp),
    Barrier,
}

impl Kind {
    /// Method name as it appears in the `Communicator` trait.
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::AllReduce(_) => "all_reduce",
            Self::Broadcast { .. } => "broadcast",
            Self::AllGather => "all_gather",
            Self::ReduceScatter(_) => "reduce_scatter",
            Self::Barrier => "barrier",
        }
    }
}

/// One rank's call. `send` and `recv` are host addresses; both equal the buffer for in-place ops.
#[derive(Clone, Copy, Debug)]
pub(super) struct Post {
    pub(super) kind: Kind,
    pub(super) send: u64,
    pub(super) recv: u64,
    pub(super) count: usize,
    pub(super) dtype: DType,
}

impl Post {
    fn same_call(&self, other: &Self) -> bool {
        self.kind == other.kind && self.count == other.count && self.dtype == other.dtype
    }

    fn describe(&self) -> String {
        format!(
            "{:?} with count {} and dtype {:?}",
            self.kind, self.count, self.dtype
        )
    }

    /// Bytes of one `count`-element block.
    fn block_bytes(&self) -> Result<usize, Fault> {
        self.count
            .checked_mul(self.dtype.size_in_bytes())
            .ok_or_else(|| Fault::Invalid {
                arg: "count",
                reason: format!(
                    "{} elements of {:?} overflow the address space",
                    self.count, self.dtype
                ),
            })
    }
}

/// Why a round failed. Every rank of the round receives the same fault.
#[derive(Clone, Debug)]
pub(super) enum Fault {
    Mismatch(String),
    Dtype { dtype: DType, op: &'static str },
    Invalid { arg: &'static str, reason: String },
}

impl Fault {
    pub(super) fn to_error(&self) -> Error {
        match self {
            Self::Mismatch(msg) => Error::Backend(msg.clone()),
            Self::Dtype { dtype, op } => Error::UnsupportedDType {
                dtype: *dtype,
                op: *op,
            },
            Self::Invalid { arg, reason } => Error::InvalidArgument {
                arg: *arg,
                reason: reason.clone(),
            },
        }
    }
}

fn overflow(world: usize) -> Fault {
    Fault::Invalid {
        arg: "count",
        reason: format!("count times world size {world} overflows the address space"),
    }
}

/// Bytes each rank reads from `send` and writes to `recv`, in that order.
fn spans(kind: Kind, block: usize, world: usize) -> Result<(usize, usize), Fault> {
    let full = block.checked_mul(world).ok_or_else(|| overflow(world))?;
    Ok(match kind {
        Kind::AllReduce(_) | Kind::Broadcast { .. } => (block, block),
        Kind::AllGather => (block, full),
        Kind::ReduceScatter(_) => (full, block),
        Kind::Barrier => (0, 0),
    })
}

/// Check that every rank made the same valid call.
fn check(posts: &[Post]) -> Result<(), Fault> {
    let Some(first) = posts.first() else {
        return Ok(());
    };
    for (rank, post) in posts.iter().enumerate().skip(1) {
        if !post.same_call(first) {
            return Err(Fault::Mismatch(format!(
                "LocalCommunicator: rank 0 called {} but rank {rank} called {}. \
                 Every rank must make the same call",
                first.describe(),
                post.describe()
            )));
        }
    }
    let world = posts.len();
    match first.kind {
        Kind::AllReduce(_) | Kind::ReduceScatter(_) if !reduce::supports(first.dtype) => {
            return Err(Fault::Dtype {
                dtype: first.dtype,
                op: first.kind.name(),
            });
        }
        Kind::Broadcast { root } if root >= world => {
            return Err(Fault::Invalid {
                arg: "root",
                reason: format!("root {root} is not a rank of a group of {world}"),
            });
        }
        _ => {}
    }
    let (read, write) = spans(first.kind, first.block_bytes()?, world)?;
    for (rank, post) in posts.iter().enumerate() {
        if (read > 0 && post.send == 0) || (write > 0 && post.recv == 0) {
            return Err(Fault::Invalid {
                arg: "ptr",
                reason: format!("rank {rank} passed a null pointer to {}", first.kind.name()),
            });
        }
    }
    Ok(())
}

/// Check the round and compute its result into `scratch`. The last rank to arrive runs this.
///
/// # Safety
///
/// `posts` holds one post per rank, in rank order.
/// Each post's `send` names the bytes its call reads, as the `Communicator` contract requires.
/// Every posting rank stays blocked, with its buffers live and unwritten, until this returns.
pub(super) unsafe fn execute(posts: &[Option<Post>], scratch: &mut Vec<u8>) -> Result<(), Fault> {
    let posts: Vec<Post> = posts.iter().flatten().copied().collect();
    check(&posts)?;
    let Some(first) = posts.first() else {
        return Ok(());
    };
    let (read, _) = spans(first.kind, first.block_bytes()?, posts.len())?;
    // SAFETY (every `host::read` and `host::read_into` below): `check` rejected null pointers for a non-empty span, and
    // the caller guarantees each `send` names `read` live bytes with no concurrent writer.
    match first.kind {
        Kind::Barrier => scratch.clear(),
        Kind::Broadcast { root } => {
            let src = posts.get(root).map_or(0, |p| p.send);
            scratch.clear();
            unsafe { host::read_into(src, read, scratch) };
        }
        Kind::AllGather => {
            scratch.clear();
            for post in &posts {
                unsafe { host::read_into(post.send, read, scratch) };
            }
        }
        Kind::AllReduce(op) | Kind::ReduceScatter(op) => {
            let inputs: Vec<Vec<u8>> = posts
                .iter()
                .map(|p| unsafe { host::read(p.send, read) })
                .collect();
            *scratch = reduce::reduce(&inputs, first.dtype, op).ok_or(Fault::Dtype {
                dtype: first.dtype,
                op: first.kind.name(),
            })?;
        }
    }
    Ok(())
}

/// Copy this rank's share of a computed round into its `recv` buffer.
///
/// # Safety
///
/// `post` passed [`execute`] for this round, and `scratch` holds that round's result.
/// `post.recv` names this rank's output bytes, as the `Communicator` contract requires.
/// Only the calling rank touches that buffer while this runs.
pub(super) unsafe fn deliver(
    rank: usize,
    world: usize,
    post: &Post,
    scratch: &[u8],
) -> Result<(), Fault> {
    let block = post.block_bytes()?;
    let out = match post.kind {
        Kind::Barrier => return Ok(()),
        Kind::Broadcast { root } if root == rank => return Ok(()),
        Kind::AllReduce(_) | Kind::Broadcast { .. } | Kind::AllGather => scratch,
        Kind::ReduceScatter(_) => {
            let start = rank.checked_mul(block).ok_or_else(|| overflow(world))?;
            scratch
                .get(start..start.saturating_add(block))
                .ok_or_else(|| Fault::Invalid {
                    arg: "count",
                    reason: format!("rank {rank} has no slice in a {world}-rank reduce_scatter"),
                })?
        }
    };
    // SAFETY: `execute` rejected a null `recv` for a non-empty span. The caller guarantees `recv`
    // names `out.len()` writable bytes that only this rank touches. `scratch` is group-owned
    // memory, so it cannot overlap a caller buffer.
    unsafe { host::write(post.recv, out) };
    Ok(())
}
