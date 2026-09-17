//! Shared discriminant for the fused activation-mul variants.

/// Which fused activation-mul variant
#[derive(Clone, Copy)]
pub(super) enum FusedKind {
    Silu,
    Gelu,
    Relu,
    Sigmoid,
}
