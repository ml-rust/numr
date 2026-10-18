//! The rendezvous every rank of a local group meets at.
//!
//! A round has two phases, tracked by `epoch`: even while ranks arrive, odd while they depart.
//! Arrival: each rank posts its call. The last rank to arrive checks and computes the round.
//! Departure: each rank copies its own result out. The last rank to depart resets the round.
//! No rank returns before every rank has departed, so no rank starts the next round early.
//! Every peer pointer is read under the lock while its owner is blocked in its call.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::plan::{self, Fault, Post};
use crate::error::{Error, Result};

struct Round {
    epoch: u64,
    arrived: usize,
    departed: usize,
    posts: Vec<Option<Post>>,
    scratch: Vec<u8>,
    fault: Option<Fault>,
}

/// State one group shares across its ranks.
pub(super) struct Shared {
    world: usize,
    timeout: Duration,
    aborted: AtomicBool,
    round: Mutex<Round>,
    cv: Condvar,
}

impl Shared {
    pub(super) fn new(world: usize, timeout: Duration) -> Self {
        Self {
            world,
            timeout,
            aborted: AtomicBool::new(false),
            round: Mutex::new(Round {
                epoch: 0,
                arrived: 0,
                departed: 0,
                posts: vec![None; world],
                scratch: Vec::new(),
                fault: None,
            }),
            cv: Condvar::new(),
        }
    }

    pub(super) fn world(&self) -> usize {
        self.world
    }

    pub(super) fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }

    /// Mark the group aborted and wake every waiting rank.
    pub(super) fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
        // Taking the lock orders this notify after any waiter's flag check, so no wakeup is lost.
        drop(self.lock());
        self.cv.notify_all();
    }

    /// Mark the group aborted while the caller holds the lock.
    fn abort_held(&self) {
        self.aborted.store(true, Ordering::SeqCst);
        self.cv.notify_all();
    }

    /// Lock the round. A poisoned lock aborts the group, and the caller then sees the abort.
    fn lock(&self) -> MutexGuard<'_, Round> {
        self.round.lock().unwrap_or_else(|poisoned| {
            self.abort_held();
            poisoned.into_inner()
        })
    }

    fn aborted_error(&self, rank: usize, name: &str) -> Error {
        Error::Backend(format!(
            "LocalCommunicator rank {rank}: {name} stopped because the group is aborted. \
             A rank called abort(), timed out, panicked under abort_on_panic(), or was dropped"
        ))
    }

    /// Block until `pending` turns false, the group aborts, or the timeout passes.
    ///
    /// A finished round wins over an abort, so a rank never reports a completed call as failed.
    /// A timeout aborts the group, so every peer wakes and returns too.
    fn wait_while<'a>(
        &self,
        mut round: MutexGuard<'a, Round>,
        rank: usize,
        name: &str,
        pending: impl Fn(&Round) -> bool,
    ) -> Result<MutexGuard<'a, Round>> {
        let deadline = Instant::now().checked_add(self.timeout);
        loop {
            if !pending(&round) {
                return Ok(round);
            }
            if self.is_aborted() {
                return Err(self.aborted_error(rank, name));
            }
            let left = match deadline {
                Some(at) => at.saturating_duration_since(Instant::now()),
                None => self.timeout,
            };
            if left.is_zero() {
                self.abort_held();
                return Err(Error::Backend(format!(
                    "LocalCommunicator rank {rank}: {name} timed out after {:?} waiting for \
                     the other {} ranks. The group is now aborted",
                    self.timeout,
                    self.world - 1
                )));
            }
            round = match self.cv.wait_timeout(round, left) {
                Ok((guard, _)) => guard,
                Err(poisoned) => {
                    self.abort_held();
                    poisoned.into_inner().0
                }
            };
        }
    }

    /// Run one collective round for `rank`.
    ///
    /// # Safety
    ///
    /// `post.send` and `post.recv` must meet the `Communicator` contract for `post.kind`.
    /// Both must stay valid until this returns.
    pub(super) unsafe fn run(&self, rank: usize, post: Post) -> Result<()> {
        let name = post.kind.name();
        let mut round = self.lock();
        if self.is_aborted() {
            return Err(self.aborted_error(rank, name));
        }
        // Every rank returns only after a reset, so an odd `epoch` or a filled slot here means
        // this rank has a second call in flight from another thread.
        let free = round.epoch.is_multiple_of(2) && matches!(round.posts.get(rank), Some(None));
        if !free {
            self.abort_held();
            return Err(Error::Backend(format!(
                "LocalCommunicator rank {rank}: {name} started while this rank had a call \
                 in flight. Use each rank from one thread at a time. The group is now aborted"
            )));
        }
        if let Some(slot) = round.posts.get_mut(rank) {
            *slot = Some(post);
        }
        round.arrived += 1;
        let arriving = round.epoch;
        if round.arrived == self.world {
            let Round {
                posts,
                scratch,
                fault,
                epoch,
                ..
            } = &mut *round;
            // SAFETY: the abort check above ran under this lock, so every other rank is blocked
            // in `wait_while` and cannot return before this guard drops. Their posted buffers
            // stay live and unwritten for the whole computation.
            *fault = unsafe { plan::execute(posts, scratch) }.err();
            *epoch += 1;
            self.cv.notify_all();
        } else {
            round = self.wait_while(round, rank, name, |r| r.epoch == arriving)?;
        }

        let outcome = match &round.fault {
            Some(fault) => Err(fault.to_error()),
            // SAFETY: the round passed `execute`, `scratch` holds its result, and the caller
            // guarantees `post.recv` is this rank's live output buffer.
            None => unsafe { plan::deliver(rank, self.world, &post, &round.scratch) }
                .map_err(|f| f.to_error()),
        };

        round.departed += 1;
        let departing = round.epoch;
        if round.departed == self.world {
            round.arrived = 0;
            round.departed = 0;
            round.posts.fill(None);
            round.fault = None;
            round.epoch += 1;
            self.cv.notify_all();
        } else {
            round = self.wait_while(round, rank, name, |r| r.epoch == departing)?;
        }
        drop(round);
        outcome
    }
}
