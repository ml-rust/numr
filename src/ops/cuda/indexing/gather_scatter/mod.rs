//! Gather, scatter, index_select, index_put and index-bounds validation for
//! the CUDA runtime.

mod gather;
mod index_put;
mod index_select;
mod scatter;
mod validate;

pub use gather::{gather, gather_2d};
pub use index_put::index_put;
pub use index_select::index_select;
pub use scatter::{scatter, slice_assign};
