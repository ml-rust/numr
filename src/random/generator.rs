//! Seed source for every random op that takes no explicit seed.
//!
//! Each unseeded op on every backend draws one `u64` from [`next_call_seed`].
//! The op seeds its kernel generator from that value.
//!
//! # Sources
//!
//! [`next_call_seed`] reads the first source that is set:
//!
//! 1. Thread override: the innermost [`with_seed`] scope on the calling thread.
//! 2. Global generator: set by [`manual_seed`] or [`set_seed_state`].
//! 3. Entropy: a per-process random hash of a counter and the clock.
//!
//! A seeded source holds a seed and a call counter.
//! Call `n` returns output `n` of the SplitMix64 stream for that seed.
//! The counter then advances by one.
//!
//! # Guarantee
//!
//! - Same seed, same backend, same op order and same shapes give bit-identical results.
//! - Different backends give different results for the same seed.
//! - Callers on several threads that share the global counter are not deterministic.
//!   Wrap each such caller in [`with_seed`].
//! - Ops with an explicit seed, such as `rand_seeded`, never read this source.
//!   They leave every counter unchanged.
//!
//! # Checkpoint resume
//!
//! [`seed_state`] returns the global `(seed, calls)` pair.
//! [`set_seed_state`] restores it, so the stream continues at the same call.
//! Both functions read and write the global generator only, never a thread override.

use std::cell::Cell;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// SplitMix64 increment: the 64-bit golden ratio.
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// A seeded generator: its seed and the number of seeds it returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GenState {
    seed: u64,
    calls: u64,
}

impl GenState {
    fn new(seed: u64) -> Self {
        Self { seed, calls: 0 }
    }

    /// Return output `calls` of the SplitMix64 stream for `seed`, then count it.
    fn advance(&mut self) -> u64 {
        let step = self.calls.wrapping_add(1).wrapping_mul(GOLDEN_GAMMA);
        self.calls = self.calls.wrapping_add(1);
        splitmix64_mix(self.seed.wrapping_add(step))
    }
}

/// Global generator. `None` means unseeded.
static GLOBAL: Mutex<Option<GenState>> = Mutex::new(None);

/// Counter for the unseeded path, so two calls in one clock tick differ.
static ENTROPY_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Stack of `with_seed` scopes on this thread. The last entry is the innermost.
    static OVERRIDES: Cell<Vec<GenState>> = const { Cell::new(Vec::new()) };
}

/// Lock the global generator.
///
/// A panic while the lock is held leaves a valid `Option<GenState>`.
/// The lock recovers it from the poison state.
fn lock_global() -> MutexGuard<'static, Option<GenState>> {
    GLOBAL.lock().unwrap_or_else(PoisonError::into_inner)
}

/// SplitMix64 output mix (Steele, Lea and Flood 2014).
#[inline]
fn splitmix64_mix(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Seed the global generator and reset its call counter to 0.
///
/// Every later unseeded random op on this process draws from this stream.
/// A [`with_seed`] scope on the calling thread still takes precedence.
pub fn manual_seed(seed: u64) {
    *lock_global() = Some(GenState::new(seed));
}

/// Return the global `(seed, calls)` pair, or `None` when unseeded.
///
/// `calls` is the number of seeds the global generator returned since it was seeded.
/// Store this value in a checkpoint and restore it with [`set_seed_state`].
pub fn seed_state() -> Option<(u64, u64)> {
    let state = *lock_global();
    state.map(|state| (state.seed, state.calls))
}

/// Restore the global generator from a [`seed_state`] value.
///
/// `Some((seed, calls))` continues the stream at call `calls`.
/// `None` returns the global generator to the unseeded entropy path.
pub fn set_seed_state(state: Option<(u64, u64)>) {
    *lock_global() = state.map(|(seed, calls)| GenState { seed, calls });
}

/// Return the seed for the next unseeded random op.
///
/// The value comes from the innermost [`with_seed`] scope on this thread.
/// Without a scope, it comes from the global generator.
/// Without a global seed, it comes from entropy and is not reproducible.
pub fn next_call_seed() -> u64 {
    if let Some(seed) = next_thread_seed() {
        return seed;
    }
    let mut global = lock_global();
    match global.as_mut() {
        Some(state) => state.advance(),
        None => {
            drop(global);
            entropy_seed()
        }
    }
}

/// Run `f` with a thread-local generator seeded with `seed`.
///
/// Every unseeded random op on this thread inside `f` draws from this generator.
/// Its call counter starts at 0 on each entry.
/// The global generator and its counter stay unchanged.
/// Other threads never see this generator.
///
/// Scopes nest. An inner scope hides the outer one until it returns.
/// The outer counter then continues where it stopped.
/// The scope ends when `f` returns or unwinds.
///
/// During thread-local destruction the scope cannot install.
/// In that case `f` draws from the global generator.
pub fn with_seed<T>(seed: u64, f: impl FnOnce() -> T) -> T {
    let _scope = ThreadSeedScope::enter(seed);
    f()
}

/// Draw from the innermost thread override, if one is set.
fn next_thread_seed() -> Option<u64> {
    OVERRIDES
        .try_with(|cell| {
            let mut stack = cell.take();
            let seed = stack.last_mut().map(GenState::advance);
            cell.set(stack);
            seed
        })
        .ok()
        .flatten()
}

/// Guard for one `with_seed` scope. Drop removes the scope it added.
struct ThreadSeedScope {
    entered: bool,
}

impl ThreadSeedScope {
    fn enter(seed: u64) -> Self {
        let entered = OVERRIDES
            .try_with(|cell| {
                let mut stack = cell.take();
                stack.push(GenState::new(seed));
                cell.set(stack);
            })
            .is_ok();
        Self { entered }
    }
}

impl Drop for ThreadSeedScope {
    fn drop(&mut self) {
        if !self.entered {
            return;
        }
        // An error here means thread-local storage is gone, so no stack is left to restore.
        let _ = OVERRIDES.try_with(|cell| {
            let mut stack = cell.take();
            stack.pop();
            cell.set(stack);
        });
    }
}

/// Unseeded path: hash a counter and the clock with per-process random keys.
///
/// `RandomState` keys come from the OS once per thread, then change on each use.
/// No extra crate is needed.
fn entropy_seed() -> u64 {
    let counter = ENTROPY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut hasher = RandomState::new().build_hasher();
    counter.hash(&mut hasher);
    clock_nanos().hash(&mut hasher);
    splitmix64_mix(
        hasher
            .finish()
            .wrapping_add(counter.wrapping_mul(GOLDEN_GAMMA)),
    )
}

/// Nanoseconds since the Unix epoch, or 0 when the clock is before it.
#[cfg(not(target_arch = "wasm32"))]
fn clock_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// `SystemTime::now` panics on `wasm32-unknown-unknown`, so wasm uses no clock.
#[cfg(target_arch = "wasm32")]
fn clock_nanos() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::ops::{ActivationOps, RandomOps};
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use crate::runtime::{Runtime, RuntimeClient};
    use crate::tensor::Tensor;

    /// Serializes the tests in this module that read or write the global generator.
    static GLOBAL_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn global_lock() -> MutexGuard<'static, ()> {
        GLOBAL_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn client() -> CpuClient {
        CpuRuntime::default_client(&CpuDevice::new())
    }

    fn draws(n: usize) -> Vec<u64> {
        (0..n).map(|_| next_call_seed()).collect()
    }

    /// Run `f` under two scopes with `seed` and one with `seed + 1`.
    /// The first two must match. The third must differ.
    fn assert_seeded<T: PartialEq + std::fmt::Debug>(seed: u64, f: impl Fn() -> T) {
        let a = with_seed(seed, &f);
        let b = with_seed(seed, &f);
        let c = with_seed(seed.wrapping_add(1), &f);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn stream_is_splitmix64_of_seed() {
        // Reference outputs of SplitMix64 seeded with 0.
        let first = with_seed(0, || draws(3));
        assert_eq!(
            first,
            vec![
                0xe220_a839_7b1d_cdaf,
                0x6e78_9e6a_a1b9_65f4,
                0x06c4_5d18_8009_454f
            ]
        );
    }

    #[test]
    fn manual_seed_repeats_global_stream() {
        let _lock = global_lock();
        manual_seed(42);
        let a = draws(8);
        assert_eq!(seed_state(), Some((42, 8)));
        manual_seed(42);
        let b = draws(8);
        manual_seed(43);
        let c = draws(8);
        set_seed_state(None);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn adjacent_seeds_do_not_share_shifted_streams() {
        let s0 = with_seed(0, || draws(16));
        let s1 = with_seed(1, || draws(16));
        assert!(s1.iter().all(|v| !s0.contains(v)));
    }

    #[test]
    fn set_seed_state_resumes_stream() {
        let _lock = global_lock();
        manual_seed(7);
        let _ = draws(5);
        let saved = seed_state();
        let tail = draws(4);
        set_seed_state(saved);
        let resumed = draws(4);
        set_seed_state(None);
        assert_eq!(saved, Some((7, 5)));
        assert_eq!(tail, resumed);
    }

    #[test]
    fn unseeded_draws_differ() {
        let _lock = global_lock();
        set_seed_state(None);
        let a = draws(4);
        assert_eq!(seed_state(), None);
        assert!(a.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn with_seed_overrides_global_without_moving_its_counter() {
        let _lock = global_lock();
        manual_seed(9);
        let global_first = draws(3);
        manual_seed(1);
        let before = seed_state();
        let scoped = with_seed(9, || draws(3));
        let after = seed_state();
        set_seed_state(None);
        assert_eq!(scoped, global_first);
        assert_eq!(before, after);
    }

    #[test]
    fn with_seed_nests_and_restores_outer_counter() {
        let flat = with_seed(1, || draws(2));
        let inner_alone = with_seed(2, || draws(2));
        let (outer, inner) = with_seed(1, || {
            let a = next_call_seed();
            let inner = with_seed(2, || draws(2));
            let b = next_call_seed();
            (vec![a, b], inner)
        });
        assert_eq!(outer, flat);
        assert_eq!(inner, inner_alone);
    }

    #[test]
    fn with_seed_scope_ends_on_unwind() {
        let expected = with_seed(3, || draws(2));
        let got = with_seed(3, || {
            let a = next_call_seed();
            let unwound = std::panic::catch_unwind(|| with_seed(4, || panic!("unwind test")));
            assert!(unwound.is_err());
            vec![a, next_call_seed()]
        });
        assert_eq!(got, expected);
    }

    #[test]
    fn threads_keep_independent_scopes() {
        let expected = with_seed(5, || draws(64));
        let handles: Vec<_> = (0..4)
            .map(|_| std::thread::spawn(|| with_seed(5, || draws(64))))
            .collect();
        for handle in handles {
            assert_eq!(handle.join().ok(), Some(expected.clone()));
        }
    }

    #[test]
    fn cpu_ops_repeat_under_seed() {
        let c = client();
        assert_seeded(11, || {
            c.rand(&[64], DType::F32).map(|t| t.to_vec::<f32>()).ok()
        });
        assert_seeded(12, || {
            c.randn(&[64], DType::F32).map(|t| t.to_vec::<f32>()).ok()
        });
        assert_seeded(13, || {
            c.randint(0, 1_000_000, &[64], DType::I64)
                .map(|t| t.to_vec::<i64>())
                .ok()
        });
        assert_seeded(14, || {
            c.bernoulli(0.5, &[64], DType::F32)
                .map(|t| t.to_vec::<f32>())
                .ok()
        });
        assert_seeded(15, || c.randperm(64).map(|t| t.to_vec::<i64>()).ok());
    }

    #[test]
    fn cpu_multinomial_repeats_under_seed() {
        let c = client();
        let probs = vec![1.0f32; 2 * 64];
        let probs = Tensor::<CpuRuntime>::from_slice(&probs, &[2, 64], c.device()).unwrap();
        assert_seeded(16, || {
            c.multinomial(&probs, 16, true)
                .map(|t| t.to_vec::<i64>())
                .ok()
        });
        assert_seeded(17, || {
            c.multinomial(&probs, 16, false)
                .map(|t| t.to_vec::<i64>())
                .ok()
        });
    }

    #[test]
    fn cpu_dropout_mask_repeats_under_seed() {
        let c = client();
        let ones = Tensor::<CpuRuntime>::ones(&[256], DType::F32, c.device()).unwrap();
        assert_seeded(18, || {
            c.dropout(&ones, 0.5, true).map(|t| t.to_vec::<f32>()).ok()
        });
    }

    #[test]
    fn cpu_rand_repeats_after_manual_seed() {
        let _lock = global_lock();
        let c = client();
        manual_seed(123);
        let a = c.rand(&[32], DType::F32).map(|t| t.to_vec::<f32>()).ok();
        manual_seed(123);
        let b = c.rand(&[32], DType::F32).map(|t| t.to_vec::<f32>()).ok();
        set_seed_state(None);
        assert!(a.is_some());
        assert_eq!(a, b);
    }
}
