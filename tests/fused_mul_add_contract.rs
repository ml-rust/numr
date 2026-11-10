//! Exact CPU fused arithmetic and conservative provider declarations.

#[path = "common/float_arithmetic_types.rs"]
mod arithmetic_types;
#[path = "fused_mul_add_contract/exact.rs"]
mod exact;
#[path = "fused_mul_add_contract/layouts.rs"]
mod layouts;
#[path = "fused_mul_add_contract/providers.rs"]
mod providers;
