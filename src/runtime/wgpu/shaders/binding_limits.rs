//! Binding size checks against device limits.
//!
//! wgpu panics when a bound range exceeds a device binding limit. These
//! helpers turn that case into `Error::BackendLimitation` before wgpu runs.

use wgpu::{BufferBinding, BufferUsages, Limits};

use crate::error::{Error, Result};

/// Operation name used in errors raised while creating bind groups.
pub(super) const BIND_GROUP_OP: &str = "create_bind_group";

/// Check one binding of `size` bytes against the limit for its buffer kind.
///
/// A buffer with `UNIFORM` usage and no `STORAGE` usage counts as a uniform
/// binding. Every other buffer counts as a storage binding.
pub(super) fn check_binding_size(
    limits: &Limits,
    operation: &'static str,
    binding: u32,
    usage: BufferUsages,
    size: u64,
) -> Result<()> {
    let uniform = usage.contains(BufferUsages::UNIFORM) && !usage.contains(BufferUsages::STORAGE);
    let (kind, limit) = if uniform {
        ("uniform", limits.max_uniform_buffer_binding_size)
    } else {
        ("storage", limits.max_storage_buffer_binding_size)
    };
    if size > limit {
        return Err(Error::BackendLimitation {
            backend: "wgpu",
            operation,
            reason: format!(
                "{kind} binding {binding} is {size} bytes, over the device limit of {limit} \
                 bytes; the tensor is too large for one binding on this device"
            ),
        });
    }
    Ok(())
}

/// Size in bytes that wgpu checks for a ranged binding.
pub(super) fn binding_range_size(binding: &BufferBinding<'_>) -> u64 {
    binding.size.map_or_else(
        || binding.buffer.size().saturating_sub(binding.offset),
        |s| s.get(),
    )
}

/// Create a bind group after checking every buffer binding against device limits.
///
/// Use this for launchers that bind buffers without a `PipelineCache`.
///
/// # Errors
///
/// `BackendLimitation` when a buffer binding exceeds the device's binding size limit.
#[cfg(feature = "sparse")]
pub(crate) fn create_checked_bind_group(
    device: &wgpu::Device,
    desc: &wgpu::BindGroupDescriptor<'_>,
) -> Result<wgpu::BindGroup> {
    let limits = device.limits();
    for entry in desc.entries {
        if let wgpu::BindingResource::Buffer(binding) = &entry.resource {
            check_binding_size(
                &limits,
                BIND_GROUP_OP,
                entry.binding,
                binding.buffer.usage(),
                binding_range_size(binding),
            )?;
        }
    }
    Ok(device.create_bind_group(desc))
}
