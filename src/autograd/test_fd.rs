//! F64 CPU finite-difference check for reverse-mode gradients.
//!
//! [`check_grad_f64`] compares the gradients `backward` returns against
//! central differences of the scalar `sum(op(x) * w)`. The probe `w` is a
//! fixed pseudo-random tensor with distinct entries. A gradient that is
//! transposed, permuted or summed into the wrong slot fails the check.

use crate::autograd::{Var, VarGradStore, backward, var_mul, var_sum};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use crate::tensor::{Tensor, TensorId};

/// Central-difference step.
const EPS: f64 = 1e-6;

/// Allowed error, relative to `max(|analytic|, |numeric|, 1)`.
const TOL: f64 = 1e-6;

/// Build a contiguous F64 CPU tensor.
pub(crate) fn f64_tensor(values: &[f64], shape: &[usize]) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(values, shape, &CpuDevice::new()).unwrap_or_else(|e| {
        panic!(
            "f64_tensor: shape {shape:?} with {} values: {e}",
            values.len()
        )
    })
}

/// Read the gradient `Var` for `id` from a second-order store.
pub(crate) fn grad_var(grads: &VarGradStore<CpuRuntime>, id: TensorId) -> Result<Var<CpuRuntime>> {
    grads
        .get_var(id)
        .map(Var::alias)
        .ok_or_else(|| Error::Internal(format!("no gradient Var for tensor {id:?}")))
}

/// Fixed probe weights: magnitudes in [0.5, 1.5), both signs, no zeros.
fn probe_weights(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            let centered = unit * 2.0 - 1.0;
            centered.signum() * (0.5 + centered.abs())
        })
        .collect()
}

/// Run `op` on fresh leaves and return `sum(op(x) * w)`.
fn objective<F>(name: &str, op: &F, inputs: &[Tensor<CpuRuntime>], w: &[f64], c: &CpuClient) -> f64
where
    F: Fn(&[Var<CpuRuntime>], &CpuClient) -> Result<Var<CpuRuntime>>,
{
    let vars: Vec<Var<CpuRuntime>> = inputs.iter().map(|t| Var::new(t.clone(), true)).collect();
    let out = op(&vars, c).unwrap_or_else(|e| panic!("{name}: forward error: {e}"));
    let values = read_f64(name, "output", out.tensor());
    assert_eq!(
        values.len(),
        w.len(),
        "{name}: output size changed from {} to {} under perturbation",
        w.len(),
        values.len()
    );
    values.iter().zip(w).map(|(v, wi)| v * wi).sum()
}

fn read_f64(name: &str, what: &str, t: &Tensor<CpuRuntime>) -> Vec<f64> {
    assert_eq!(
        t.dtype(),
        DType::F64,
        "{name}: {what} must be F64, got {:?}",
        t.dtype()
    );
    t.contiguous()
        .unwrap_or_else(|e| panic!("{name}: {what} contiguous error: {e}"))
        .to_vec::<f64>()
}

/// Check the analytic gradient of every input of `op` against central differences.
///
/// `op` receives one leaf per entry of `inputs`, each with `requires_grad == true`.
/// Values the check must not differentiate, such as an index tensor, belong in
/// the closure. Every input must be an F64 CPU tensor.
///
/// The scalar is `sum(op(x) * w)` with a fixed probe `w`. The step is `1e-6`.
/// Each element must satisfy `|a - n| <= 1e-6 * max(|a|, |n|, 1)`.
///
/// A missing gradient counts as zero. The check panics with the op name, the
/// input number, the flat element index, and both values.
pub(crate) fn check_grad_f64<F>(name: &str, op: F, inputs: &[Tensor<CpuRuntime>])
where
    F: Fn(&[Var<CpuRuntime>], &CpuClient) -> Result<Var<CpuRuntime>>,
{
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let vars: Vec<Var<CpuRuntime>> = inputs.iter().map(|t| Var::new(t.clone(), true)).collect();
    let out = op(&vars, &client).unwrap_or_else(|e| panic!("{name}: forward error: {e}"));
    assert!(
        out.requires_grad(),
        "{name}: output does not require grad, so no input reaches it"
    );

    let w_values = probe_weights(out.numel());
    let w = Var::new(f64_tensor(&w_values, out.shape()), false);
    let loss = var_mul(&out, &w, &client)
        .and_then(|weighted| var_sum(&weighted, &[], false, &client))
        .unwrap_or_else(|e| panic!("{name}: probe loss error: {e}"));
    let grads = backward(&loss, &client).unwrap_or_else(|e| panic!("{name}: backward error: {e}"));

    for (i, (var, input)) in vars.iter().zip(inputs).enumerate() {
        let base = read_f64(name, &format!("input {i}"), input);
        let analytic = match grads.get(var.id()) {
            Some(g) => read_f64(name, &format!("gradient of input {i}"), g),
            None => vec![0.0; base.len()],
        };
        assert_eq!(
            analytic.len(),
            base.len(),
            "{name}: input {i} has {} elements but its gradient has {}",
            base.len(),
            analytic.len()
        );

        for (j, (&x0, &a)) in base.iter().zip(&analytic).enumerate() {
            let mut shifted: Vec<Tensor<CpuRuntime>> = inputs.to_vec();
            let mut probe = base.clone();

            probe[j] = x0 + EPS;
            shifted[i] = f64_tensor(&probe, input.shape());
            let f_plus = objective(name, &op, &shifted, &w_values, &client);

            probe[j] = x0 - EPS;
            shifted[i] = f64_tensor(&probe, input.shape());
            let f_minus = objective(name, &op, &shifted, &w_values, &client);

            let numeric = (f_plus - f_minus) / (2.0 * EPS);
            let scale = a.abs().max(numeric.abs()).max(1.0);
            assert!(
                (a - numeric).abs() <= TOL * scale,
                "{name}: input {i} element {j}: analytic {a:.12e}, numeric {numeric:.12e}, \
                 error {:.3e} exceeds {:.3e}",
                (a - numeric).abs(),
                TOL * scale
            );
        }
    }
}
