//! Operation and format selectors for the CUDA sparse merge kernels.

/// Element-wise operation of a sparse merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMergeOp {
    /// `a + b` over the union of both patterns.
    Add,
    /// `a - b` over the union of both patterns.
    Sub,
    /// `a * b` over the intersection of both patterns.
    Mul,
    /// `a / b` over the intersection of both patterns.
    Div,
}

impl SparseMergeOp {
    /// The operation name inside the kernel names: `add`, `sub`, `mul`, `div`.
    pub fn kernel_prefix(&self) -> &'static str {
        match self {
            SparseMergeOp::Add => "add",
            SparseMergeOp::Sub => "sub",
            SparseMergeOp::Mul => "mul",
            SparseMergeOp::Div => "div",
        }
    }

    /// Whether the output pattern is the union of both patterns (add, sub).
    pub fn is_union(&self) -> bool {
        matches!(self, SparseMergeOp::Add | SparseMergeOp::Sub)
    }

    /// Whether the output pattern is the intersection of both patterns (mul, div).
    pub fn is_intersection(&self) -> bool {
        !self.is_union()
    }
}

/// Sparse storage format of a merge operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Row pointers + column indices.
    Csr,
    /// Column pointers + row indices.
    Csc,
    /// Row indices + column indices.
    Coo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_and_intersection_split_the_ops() {
        assert!(SparseMergeOp::Add.is_union());
        assert!(SparseMergeOp::Sub.is_union());
        assert!(SparseMergeOp::Mul.is_intersection());
        assert!(SparseMergeOp::Div.is_intersection());
        assert_eq!(SparseMergeOp::Div.kernel_prefix(), "div");
    }
}
