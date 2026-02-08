//! Operations on Var that build the computation graph
//!
//! These functions perform forward computation and create the appropriate
//! backward functions for gradient tracking.
//!
//! # Example
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, var_mul, backward};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! // Create leaf variables
//! let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
//! let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
//!
//! // Build computation graph: z = x * y
//! let z = var_mul(&x, &y, &client)?;
//!
//! // Compute gradients
//! let grads = backward(&z, &client)?;
//! # Ok::<(), numr::error::Error>(())
//! ```

use super::Var;
use super::ops::*;
use crate::algorithm::LinearAlgebraAlgorithms;
use crate::error::Result;
use crate::ops::{CompareOps, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

// ============================================================================
// Macro-based code generation for var_* functions
// ============================================================================
//
// These macros eliminate boilerplate by generating var_* functions with
// consistent patterns. Each macro handles a specific backward constructor
// signature pattern.

/// Binary operation with shape-based backward (add, sub)
macro_rules! impl_var_binary_op_shapes {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor(), b.tensor())?;

            if a.requires_grad() || b.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(
                    a.id(),
                    b.id(),
                    a.shape(),
                    b.shape(),
                    a.grad_fn().cloned(),
                    b.grad_fn().cloned(),
                );
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Binary operation with tensor-based backward (mul, div, matmul)
macro_rules! impl_var_binary_op_tensors {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor(), b.tensor())?;

            if a.requires_grad() || b.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(
                    a.id(),
                    b.id(),
                    a.tensor().clone(),
                    b.tensor().clone(),
                    a.grad_fn().cloned(),
                    b.grad_fn().cloned(),
                );
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with id-only backward (neg)
macro_rules! impl_var_unary_op_id {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with output-based backward (exp, sqrt, tanh, recip, sigmoid)
macro_rules! impl_var_unary_op_output {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), output.clone());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with input tensor backward (log, sin, cos, square, abs, tan, relu)
macro_rules! impl_var_unary_op_input {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), a.tensor().clone());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Scalar operation with id-only backward (add_scalar, sub_scalar)
macro_rules! impl_var_scalar_op_id {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + ScalarOps<R>,
            R::Client: ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor(), scalar)?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), a.grad_fn().cloned());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Scalar operation with scalar value backward (mul_scalar, div_scalar)
macro_rules! impl_var_scalar_op_scalar {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + ScalarOps<R>,
            R::Client: ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor(), scalar)?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), scalar, a.grad_fn().cloned());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with output-based backward that requires ScalarOps (sqrt, tanh, recip, sigmoid)
///
/// These operations save the output tensor and require ScalarOps for backward computation.
macro_rules! impl_var_unary_op_output_scalar {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R> + ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), output.clone());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with input-based backward that requires ScalarOps (square, tan)
///
/// These operations save the input tensor and require ScalarOps for backward computation.
macro_rules! impl_var_unary_op_input_scalar {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R> + ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), a.tensor().clone());
                Ok(Var::from_op(output, Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

// ============================================================================
// Binary Operations
// ============================================================================

impl_var_binary_op_shapes!(
    /// Add two variables: z = a + b
    ///
    /// Creates AddBackward for gradient computation.
    var_add, add, AddBackward
);

impl_var_binary_op_shapes!(
    /// Subtract two variables: z = a - b
    ///
    /// Creates SubBackward for gradient computation.
    var_sub, sub, SubBackward
);

impl_var_binary_op_tensors!(
    /// Multiply two variables: z = a * b
    ///
    /// Creates MulBackward for gradient computation.
    var_mul, mul, MulBackward
);

impl_var_binary_op_tensors!(
    /// Divide two variables: z = a / b
    ///
    /// Creates DivBackward for gradient computation.
    var_div, div, DivBackward
);

/// Power: z = a^b
///
/// Creates PowBackward for gradient computation.
/// Note: Has special backward constructor requiring output tensor.
pub fn var_pow<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.pow(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = PowBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            b.tensor().clone(),
            output.clone(),
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

impl_var_binary_op_tensors!(
    /// Matrix multiplication: z = a @ b
    ///
    /// Creates MatmulBackward for gradient computation.
    var_matmul, matmul, MatmulBackward
);

// ============================================================================
// Unary Operations
// ============================================================================

impl_var_unary_op_id!(
    /// Negation: z = -a
    var_neg, neg, NegBackward
);

impl_var_unary_op_output!(
    /// Exponential: z = exp(a)
    var_exp, exp, ExpBackward
);

impl_var_unary_op_input!(
    /// Natural logarithm: z = log(a)
    var_log, log, LogBackward
);

impl_var_unary_op_output_scalar!(
    /// Square root: z = sqrt(a)
    ///
    /// Requires ScalarOps bound for backward computation.
    var_sqrt, sqrt, SqrtBackward
);

impl_var_unary_op_input!(
    /// Sine: z = sin(a)
    var_sin, sin, SinBackward
);

impl_var_unary_op_input!(
    /// Cosine: z = cos(a)
    var_cos, cos, CosBackward
);

impl_var_unary_op_output_scalar!(
    /// Hyperbolic tangent: z = tanh(a)
    ///
    /// Requires ScalarOps bound for backward computation.
    var_tanh, tanh, TanhBackward
);

impl_var_unary_op_input_scalar!(
    /// Square: z = a²
    ///
    /// Requires ScalarOps bound for backward computation.
    var_square, square, SquareBackward
);

impl_var_unary_op_output_scalar!(
    /// Reciprocal: z = 1/a
    ///
    /// Requires ScalarOps bound for backward computation.
    var_recip, recip, RecipBackward
);

impl_var_unary_op_input!(
    /// Absolute value: z = |a|
    var_abs, abs, AbsBackward
);

impl_var_unary_op_input_scalar!(
    /// Tangent: z = tan(a)
    ///
    /// Requires ScalarOps bound for backward computation.
    var_tan, tan, TanBackward
);

// ============================================================================
// Scalar Operations
// ============================================================================

impl_var_scalar_op_id!(
    /// Add scalar: z = a + scalar
    var_add_scalar, add_scalar, AddScalarBackward
);

impl_var_scalar_op_id!(
    /// Subtract scalar: z = a - scalar
    var_sub_scalar, sub_scalar, SubScalarBackward
);

impl_var_scalar_op_scalar!(
    /// Multiply by scalar: z = a * scalar
    var_mul_scalar, mul_scalar, MulScalarBackward
);

impl_var_scalar_op_scalar!(
    /// Divide by scalar: z = a / scalar
    var_div_scalar, div_scalar, DivScalarBackward
);

/// Power by scalar: z = a^scalar
///
/// Note: Requires both ScalarOps and TensorOps bounds.
pub fn var_pow_scalar<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.pow_scalar(a.tensor(), scalar)?;

    if a.requires_grad() {
        let grad_fn =
            PowScalarBackward::<R>::new(a.id(), a.tensor().clone(), scalar, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum along dimensions: z = sum(a, dims)
pub fn var_sum<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let output = client.sum(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = SumBackward::<R>::new(a.id(), a.shape(), dims, keepdim, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Mean along dimensions: z = mean(a, dims)
pub fn var_mean<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: ScalarOps<R>,
{
    let output = client.mean(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn =
            MeanBackward::<R>::new(a.id(), a.shape(), dims, keepdim, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Max along dimensions: z = max(a, dims)
pub fn var_max<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    let output = client.max(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = MaxBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            dims,
            keepdim,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Min along dimensions: z = min(a, dims)
pub fn var_min<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    let output = client.min(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = MinBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            dims,
            keepdim,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Statistical Operations
// ============================================================================

/// Variance along dimensions: z = var(a, dims, correction)
///
/// Creates VarBackward for gradient computation.
pub fn var_var<R, C>(
    a: &Var<R>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.var(a.tensor(), dims, keepdim, correction)?;

    if a.requires_grad() {
        let grad_fn = VarBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            dims,
            keepdim,
            correction,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Standard deviation along dimensions: z = std(a, dims, correction)
///
/// Creates StdBackward for gradient computation.
pub fn var_std<R, C>(
    a: &Var<R>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.std(a.tensor(), dims, keepdim, correction)?;

    if a.requires_grad() {
        let grad_fn = StdBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            output.clone(),
            dims,
            keepdim,
            correction,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Utility Operations
// ============================================================================

/// Clamp values to range: z = clamp(a, min, max)
///
/// Creates ClampBackward for gradient computation.
pub fn var_clamp<R, C>(a: &Var<R>, min_val: f64, max_val: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    let output = client.clamp(a.tensor(), min_val, max_val)?;

    if a.requires_grad() {
        let grad_fn = ClampBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            min_val,
            max_val,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Activation Operations
// ============================================================================

/// ReLU: z = max(0, a)
pub fn var_relu<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + CompareOps<R>,
{
    let output = client.relu(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = ReluBackward::<R>::new(a.id(), a.tensor().clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Sigmoid: z = 1 / (1 + exp(-a))
pub fn var_sigmoid<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.sigmoid(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = SigmoidBackward::<R>::new(a.id(), output.clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Softmax along dimension: z_i = exp(a_i) / sum(exp(a))
pub fn var_softmax<R, C>(a: &Var<R>, dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let output = client.softmax(a.tensor(), dim)?;

    if a.requires_grad() {
        let grad_fn = SoftmaxBackward::<R>::new(a.id(), output.clone(), dim, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Linear Algebra Operations
// ============================================================================

/// Matrix trace: scalar = tr(A)
///
/// Creates TraceBackward for gradient computation.
pub fn var_trace<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    let output = client.trace(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = TraceBackward::<R>::new(a.id(), a.tensor().clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Matrix inverse: B = A^{-1}
///
/// Creates InverseBackward for gradient computation.
pub fn var_inverse<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R>,
{
    let output = client.inverse(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = InverseBackward::<R>::new(a.id(), output.clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Matrix determinant: scalar = det(A)
///
/// Creates DetBackward for gradient computation.
pub fn var_det<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    let output = client.det(a.tensor())?;

    if a.requires_grad() {
        // Save the output tensor for the backward pass (preserves dtype)
        let grad_fn = DetBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            output.clone(), // Save output tensor instead of f64 value
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Linear system solve: x = solve(A, b) where Ax = b
///
/// Creates SolveBackward for gradient computation.
pub fn var_solve<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R> + LinearAlgebraAlgorithms<R>,
{
    let output = client.solve(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = SolveBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            output.clone(),
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Cholesky decomposition: L = cholesky(A) where A = L @ L^T
///
/// Creates CholeskyBackward for gradient computation.
pub fn var_cholesky<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    let decomp = client.cholesky_decompose(a.tensor())?;
    let output = decomp.l;

    if a.requires_grad() {
        let grad_fn = CholeskyBackward::<R>::new(a.id(), output.clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_mul_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let z = var_mul(&x, &y, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![6.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 3.0).abs() < 1e-6);
        assert!((grad_y[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_add_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let z = var_add(&x, &y, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 1.0).abs() < 1e-6);
        assert!((grad_y[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_chain_rule() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // z = (x * y) + x = x*(y+1)
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let xy = var_mul(&x, &y, &client).unwrap();
        let z = var_add(&xy, &x, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![8.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 4.0).abs() < 1e-6); // dz/dx = y + 1 = 4
        assert!((grad_y[0] - 2.0).abs() < 1e-6); // dz/dy = x = 2
    }

    #[test]
    fn test_var_exp_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device),
            true,
        );

        let z = var_exp(&x, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_matmul_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2, 1], &device),
            true,
        );

        let c = var_matmul(&a, &b, &client).unwrap();

        let c_data: Vec<f32> = c.tensor().to_vec();
        assert_eq!(c_data, vec![11.0]);

        let grads = backward(&c, &client).unwrap();

        let grad_a: Vec<f32> = grads.get(a.id()).unwrap().to_vec();
        assert_eq!(grad_a, vec![3.0, 4.0]);

        let grad_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();
        assert_eq!(grad_b, vec![1.0, 2.0]);
    }

    #[test]
    fn test_var_tan_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device),
            true,
        );

        let z = var_tan(&x, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_add_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = var_add_scalar(&x, 5.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![7.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_mul_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = var_mul_scalar(&x, 3.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![6.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_pow_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let z = var_pow_scalar(&x, 2.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![9.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_trace_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[1, 2], [3, 4]], tr(A) = 5
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device),
            true,
        );

        let trace = var_trace(&a, &client).unwrap();
        let grads = backward(&trace, &client).unwrap();

        // dL/dA = I (identity matrix)
        let grad_a: Vec<f64> = grads.get(a.id()).unwrap().to_vec();
        assert!((grad_a[0] - 1.0).abs() < 1e-6); // (0,0)
        assert!((grad_a[1] - 0.0).abs() < 1e-6); // (0,1)
        assert!((grad_a[2] - 0.0).abs() < 1e-6); // (1,0)
        assert!((grad_a[3] - 1.0).abs() < 1e-6); // (1,1)
    }

    #[test]
    fn test_var_inverse_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]]
        // A^{-1} = [[2/3, -1/3], [-1/3, 2/3]]
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device),
            true,
        );

        let inv_a = var_inverse(&a, &client).unwrap();
        // Reduce to scalar for backward - sum over all dimensions
        let loss = var_sum(&inv_a, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        // Verify gradient has correct shape
        let grad_a = grads.get(a.id()).unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);

        // Analytical gradient: dL/dA = -B^T @ dL/dB @ B^T where B = A^{-1}
        // With dL/dB = ones [2,2]:
        // B^T @ ones = [[1/3, 1/3], [1/3, 1/3]]
        // [[1/3, 1/3], [1/3, 1/3]] @ B^T = [[1/9, 1/9], [1/9, 1/9]]
        // After negation: dL/dA = [[-1/9, -1/9], [-1/9, -1/9]]
        let grad_a_data: Vec<f64> = grad_a.to_vec();
        let expected = -1.0 / 9.0;
        assert!(
            (grad_a_data[0] - expected).abs() < 1e-6,
            "grad[0,0] = {}, expected {}",
            grad_a_data[0],
            expected
        );
        assert!(
            (grad_a_data[1] - expected).abs() < 1e-6,
            "grad[0,1] = {}, expected {}",
            grad_a_data[1],
            expected
        );
        assert!(
            (grad_a_data[2] - expected).abs() < 1e-6,
            "grad[1,0] = {}, expected {}",
            grad_a_data[2],
            expected
        );
        assert!(
            (grad_a_data[3] - expected).abs() < 1e-6,
            "grad[1,1] = {}, expected {}",
            grad_a_data[3],
            expected
        );
    }

    #[test]
    fn test_var_det_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]], det(A) = 3
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device),
            true,
        );

        let det = var_det(&a, &client).unwrap();
        let grads = backward(&det, &client).unwrap();

        // dL/dA = det(A) * A^{-T} = 3 * [[2/3, -1/3], [-1/3, 2/3]] = [[2, -1], [-1, 2]]
        let grad_a: Vec<f64> = grads.get(a.id()).unwrap().to_vec();
        assert!((grad_a[0] - 2.0).abs() < 1e-6);
        assert!((grad_a[1] - (-1.0)).abs() < 1e-6);
        assert!((grad_a[2] - (-1.0)).abs() < 1e-6);
        assert!((grad_a[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_solve_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]], b = [[3], [3]]
        // x = solve(A, b) = [[1], [1]]
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f64, 3.0], &[2, 1], &device),
            true,
        );

        let x = var_solve(&a, &b, &client).unwrap();
        // Reduce to scalar for backward - sum over all dimensions
        let loss = var_sum(&x, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        // Verify gradients have correct shapes
        let grad_a = grads.get(a.id()).unwrap();
        let grad_b = grads.get(b.id()).unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_b.shape(), &[2, 1]);

        // dL/db = solve(A^T, dL/dx)
        // For symmetric A and dL/dx = [[1], [1]]: v = A^{-1} @ [[1], [1]] = [[1/3], [1/3]]
        let grad_b_data: Vec<f64> = grad_b.to_vec();
        assert!(
            (grad_b_data[0] - 1.0 / 3.0).abs() < 1e-6,
            "grad_b[0] = {}, expected {}",
            grad_b_data[0],
            1.0 / 3.0
        );
        assert!(
            (grad_b_data[1] - 1.0 / 3.0).abs() < 1e-6,
            "grad_b[1] = {}, expected {}",
            grad_b_data[1],
            1.0 / 3.0
        );

        // dL/dA = -v @ x^T where x = [[1], [1]]
        // = -[[1/3], [1/3]] @ [[1, 1]] = -[[1/3, 1/3], [1/3, 1/3]]
        let grad_a_data: Vec<f64> = grad_a.to_vec();
        let expected_a = -1.0 / 3.0;
        assert!(
            (grad_a_data[0] - expected_a).abs() < 1e-6,
            "grad_a[0,0] = {}, expected {}",
            grad_a_data[0],
            expected_a
        );
        assert!(
            (grad_a_data[1] - expected_a).abs() < 1e-6,
            "grad_a[0,1] = {}, expected {}",
            grad_a_data[1],
            expected_a
        );
        assert!(
            (grad_a_data[2] - expected_a).abs() < 1e-6,
            "grad_a[1,0] = {}, expected {}",
            grad_a_data[2],
            expected_a
        );
        assert!(
            (grad_a_data[3] - expected_a).abs() < 1e-6,
            "grad_a[1,1] = {}, expected {}",
            grad_a_data[3],
            expected_a
        );
    }

    #[test]
    fn test_var_cholesky_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[4, 2], [2, 5]] (positive definite)
        let a_data = [4.0f64, 2.0, 2.0, 5.0];
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device),
            true,
        );

        let l = var_cholesky(&a, &client).unwrap();
        // Reduce to scalar for backward - sum over all dimensions
        let loss = var_sum(&l, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        // Verify gradient has correct shape
        let grad_a = grads.get(a.id()).unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);

        // Gradient should be symmetric (since A is symmetric)
        let grad_a_data: Vec<f64> = grad_a.to_vec();
        assert!(
            (grad_a_data[1] - grad_a_data[2]).abs() < 1e-10,
            "grad_a[0,1] = {}, grad_a[1,0] = {}",
            grad_a_data[1],
            grad_a_data[2]
        );

        // Verify gradient using finite differences
        let eps = 1e-5;
        for idx in 0..4 {
            // Perturb A[idx] by +eps and -eps (symmetric perturbation)
            let mut a_plus = a_data;
            let mut a_minus = a_data;
            a_plus[idx] += eps;
            a_minus[idx] -= eps;

            // For symmetric matrix, also perturb transpose element
            let i = idx / 2;
            let j = idx % 2;
            if i != j {
                let t_idx = j * 2 + i;
                a_plus[t_idx] += eps;
                a_minus[t_idx] -= eps;
            }

            // Compute loss for perturbed matrices
            let a_plus_t = Tensor::<CpuRuntime>::from_slice(&a_plus, &[2, 2], &device);
            let a_minus_t = Tensor::<CpuRuntime>::from_slice(&a_minus, &[2, 2], &device);

            let l_plus = client.cholesky_decompose(&a_plus_t).unwrap().l;
            let l_minus = client.cholesky_decompose(&a_minus_t).unwrap().l;

            let loss_plus: f64 = l_plus.to_vec::<f64>().iter().sum();
            let loss_minus: f64 = l_minus.to_vec::<f64>().iter().sum();

            // Finite difference gradient
            let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

            // For off-diagonal elements, gradient applies to both (i,j) and (j,i)
            let expected_grad = if i != j {
                grad_a_data[idx] * 2.0 // Both elements contribute
            } else {
                grad_a_data[idx]
            };

            assert!(
                (fd_grad - expected_grad).abs() < 1e-4,
                "Finite diff gradient at ({},{}) = {}, analytical = {}",
                i,
                j,
                fd_grad,
                expected_grad
            );
        }
    }
}
