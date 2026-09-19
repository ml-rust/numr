//! Run a schedule probe once per device and key, then serve the cached pick.

use std::sync::OnceLock;

use super::cache;
use crate::error::Result;

/// Environment variable that turns tuning off. Unset, or any value other
/// than `0`, `false` or `off`, leaves it on.
pub const ENV_VAR: &str = "NUMR_CUDA_TUNE";

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether runtime tuning is on for this process.
///
/// Reads `NUMR_CUDA_TUNE` once. Unset, or any value other than `0`,
/// `false` or `off` (case-insensitive, trimmed), means on.
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| enabled_from(std::env::var(ENV_VAR).ok().as_deref()))
}

/// Parse the tuning switch from the raw environment value.
fn enabled_from(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return true;
    };
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "off"
    )
}

/// Pick a schedule value for `key` on device `device_index`.
///
/// # Contract
///
/// Every alternative the probe compares MUST produce identical bits for the
/// same inputs. The call site cites the invariance test that proves it. The
/// pick is speed only. `fallback` is the constant measured on one part; it
/// is what a build without tuning uses.
///
/// # Order
///
/// - Tuning disabled (`enabled()` is false): return `fallback`, no cache write.
/// - Cache hit: return the cached value.
/// - Cache miss: run `probe`. On `Ok(v)`, cache and return `v`. On `Err`, log
///   to stderr and return `fallback` without caching, so the next call
///   retries a transient failure.
pub fn tuned<T: Copy + Send + Sync + 'static>(
    device_index: usize,
    key: &'static str,
    fallback: T,
    probe: impl FnOnce() -> Result<T>,
) -> T {
    tuned_with(enabled(), device_index, key, fallback, probe)
}

/// `tuned` with the enable switch passed in, so tests cover both paths
/// without touching the process-wide `OnceLock`.
fn tuned_with<T: Copy + Send + Sync + 'static>(
    enabled: bool,
    device_index: usize,
    key: &'static str,
    fallback: T,
    probe: impl FnOnce() -> Result<T>,
) -> T {
    if !enabled {
        return fallback;
    }
    if let Some(value) = cache::get::<T>(device_index, key) {
        return value;
    }
    match probe() {
        Ok(value) => {
            cache::insert(device_index, key, value);
            value
        }
        Err(e) => {
            eprintln!(
                "[numr::cuda] tune probe '{key}' on device {device_index} failed, using fallback: {e}"
            );
            fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use std::cell::Cell;

    /// Device indices no other test in this crate touches.
    const DEV: usize = 9_101;

    #[test]
    fn parse_unset_is_enabled() {
        assert!(enabled_from(None));
    }

    #[test]
    fn parse_off_values_disable() {
        for value in ["0", "false", "off", " OFF ", "False"] {
            assert!(!enabled_from(Some(value)), "{value:?}");
        }
    }

    #[test]
    fn parse_other_values_enable() {
        for value in ["1", "true", "on", "", "yes"] {
            assert!(enabled_from(Some(value)), "{value:?}");
        }
    }

    #[test]
    fn probe_runs_once_per_key() {
        let calls = Cell::new(0u32);
        let probe = || {
            calls.set(calls.get() + 1);
            Ok(42u32)
        };
        assert_eq!(tuned_with(true, DEV, "once.runs_once", 0u32, probe), 42);
        assert_eq!(tuned_with(true, DEV, "once.runs_once", 0u32, probe), 42);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn err_returns_fallback_and_retries() {
        let calls = Cell::new(0u32);
        let failing = || {
            calls.set(calls.get() + 1);
            Err::<u32, _>(Error::Msg("probe failed".into()))
        };
        assert_eq!(tuned_with(true, DEV, "once.retry", 5u32, failing), 5);
        assert_eq!(tuned_with(true, DEV, "once.retry", 5u32, failing), 5);
        assert_eq!(calls.get(), 2, "a failed probe must not be cached");
        assert_eq!(cache::get::<u32>(DEV, "once.retry"), None);
        assert_eq!(tuned_with(true, DEV, "once.retry", 5u32, || Ok(9u32)), 9);
        assert_eq!(tuned_with(true, DEV, "once.retry", 5u32, failing), 9);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn disabled_never_calls_probe() {
        let calls = Cell::new(0u32);
        let probe = || {
            calls.set(calls.get() + 1);
            Ok(1u32)
        };
        assert_eq!(tuned_with(false, DEV, "once.disabled", 3u32, probe), 3);
        assert_eq!(calls.get(), 0);
        assert_eq!(cache::get::<u32>(DEV, "once.disabled"), None);
    }
}
