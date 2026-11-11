//! Provider declarations and exact square-root arithmetic under the required active environment.

#[path = "common/float_arithmetic_types.rs"]
mod arithmetic_types;
#[path = "float_arithmetic_contract/providers.rs"]
mod providers;
#[path = "float_arithmetic_contract/square_root_exact.rs"]
mod square_root_exact;
#[path = "float_arithmetic_contract/square_root_providers.rs"]
mod square_root_providers;

#[path = "float_arithmetic_contract/square_root_f32.rs"]
mod square_root_f32;
