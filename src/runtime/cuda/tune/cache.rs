//! Process-wide cache of tuned schedule values, keyed by device and name.
//!
//! Every value is a `Copy` type stored type-erased. A hit takes the read
//! lock, an insert takes the write lock. Two racing inserts for one key are
//! last-write-wins: every candidate a probe compares is bit-identical, so
//! either winner is correct.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// A tuned value's address: device index, then the call site's name.
type Key = (usize, &'static str);

type Map = HashMap<Key, Box<dyn Any + Send + Sync>>;

static CACHE: OnceLock<RwLock<Map>> = OnceLock::new();

fn map() -> &'static RwLock<Map> {
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Read the tuned value for `key` on device `device_index`.
///
/// Returns `None` on a miss, and on a hit whose stored type is not `T`.
/// A poisoned lock is read through: a value is either fully inserted or
/// absent, so a panic elsewhere cannot leave a half-written entry.
pub fn get<T: Copy + 'static>(device_index: usize, key: &'static str) -> Option<T> {
    let guard = match map().read() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard
        .get(&(device_index, key))
        .and_then(|value| value.downcast_ref::<T>())
        .copied()
}

/// Store `value` for `key` on device `device_index`, replacing any entry.
pub fn insert<T: Copy + Send + Sync + 'static>(device_index: usize, key: &'static str, value: T) {
    let mut guard = match map().write() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.insert((device_index, key), Box::new(value));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Device indices no other test in this crate touches, so the shared
    /// static cache cannot leak state between tests.
    const DEV_A: usize = 9_001;
    const DEV_B: usize = 9_002;

    #[test]
    fn roundtrip() {
        assert_eq!(get::<u32>(DEV_A, "cache.roundtrip"), None);
        insert(DEV_A, "cache.roundtrip", 7u32);
        assert_eq!(get::<u32>(DEV_A, "cache.roundtrip"), Some(7));
    }

    #[test]
    fn overwrite_is_last_write_wins() {
        insert(DEV_A, "cache.overwrite", 1u8);
        insert(DEV_A, "cache.overwrite", 2u8);
        assert_eq!(get::<u8>(DEV_A, "cache.overwrite"), Some(2));
    }

    #[test]
    fn keys_do_not_collide() {
        insert(DEV_A, "cache.key_one", 1usize);
        insert(DEV_A, "cache.key_two", 2usize);
        assert_eq!(get::<usize>(DEV_A, "cache.key_one"), Some(1));
        assert_eq!(get::<usize>(DEV_A, "cache.key_two"), Some(2));
    }

    #[test]
    fn devices_do_not_collide() {
        insert(DEV_A, "cache.device", 10i32);
        insert(DEV_B, "cache.device", 20i32);
        assert_eq!(get::<i32>(DEV_A, "cache.device"), Some(10));
        assert_eq!(get::<i32>(DEV_B, "cache.device"), Some(20));
    }

    #[test]
    fn wrong_type_is_a_miss() {
        insert(DEV_A, "cache.typed", 3u64);
        assert_eq!(get::<u32>(DEV_A, "cache.typed"), None);
        assert_eq!(get::<u64>(DEV_A, "cache.typed"), Some(3));
    }
}
