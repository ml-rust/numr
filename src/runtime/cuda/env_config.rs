//! Shared parsing helpers for CUDA runtime tuning environment variables.

/// Resolve a byte-valued tuning knob from an environment variable whose value is
/// expressed in MiB, falling back to `default_bytes` when unset or unparseable.
///
/// Used by the allocator (free-list cap) and the client (memory-pool release
/// threshold) so the MiB→bytes conversion lives in exactly one place.
pub(super) fn env_mib_to_bytes(var: &str, default_bytes: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|mib| mib.saturating_mul(1024 * 1024))
        .unwrap_or(default_bytes)
}
