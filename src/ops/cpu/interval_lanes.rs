//! CPU lane operands and lane loops shared by the interval and expansion kernels.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::IntervalOperand;
use crate::ops::impl_generic::interval::{LaneOperand, lane_operand};
use crate::runtime::cpu::kernels::interval::drive::{Input, Output, dense_get, raise, run_lanes};
use crate::runtime::cpu::kernels::interval::lane_float::{LaneFloat, positive as lane_positive};
use crate::runtime::cpu::kernels::interval::lane_interval::{Span, sqrt_nonnegative};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

type CpuTensor = Tensor<CpuRuntime>;

/// Prepare a CPU lane operand, copying general broadcast views.
pub(super) fn cpu_operand<'a>(
    value: &'a CpuTensor,
    shape: &[usize],
) -> Result<LaneOperand<'a, CpuRuntime>> {
    lane_operand(value, shape, |view| view.contiguous())
}

pub(super) fn input(operand: &LaneOperand<'_, CpuRuntime>) -> Input {
    Input::new(operand.addr(), operand.step)
}

/// Flag invalid bounds, and with `positive` also lower endpoints that are not strictly positive.
pub(super) fn flag_spans<T: LaneFloat>(
    client: &CpuClient,
    lower: Input,
    upper: Input,
    len: usize,
    positive: bool,
) -> bool {
    run_lanes(client, len, |i| {
        // Both inputs address `len` lanes.
        let span = unsafe {
            Span {
                lo: lower.get::<T>(i),
                hi: upper.get::<T>(i),
            }
        };
        span.invalid() || (positive && !lane_positive(span.lo))
    })
}

/// Flag one interval on its own shape.
pub(super) fn flag_interval(
    client: &CpuClient,
    dtype: DType,
    value: IntervalOperand<'_, CpuRuntime>,
    positive: bool,
    status: &CpuTensor,
) -> Result<()> {
    let len = value.lower.numel();
    if len == 0 {
        return Ok(());
    }
    let lower = cpu_operand(value.lower, value.lower.shape())?;
    let upper = cpu_operand(value.upper, value.upper.shape())?;
    let (lower, upper) = (input(&lower), input(&upper));
    let flagged = match dtype {
        DType::F32 => flag_spans::<f32>(client, lower, upper, len, positive),
        _ => flag_spans::<f64>(client, lower, upper, len, positive),
    };
    raise(status, flagged);
    Ok(())
}

/// Copy endpoints lane by lane and flag invalid bounds. A single output serves equal views.
pub(super) fn capture_lanes<T: LaneFloat>(
    client: &CpuClient,
    inputs: [Input; 2],
    outputs: [Output; 2],
    len: usize,
    same: bool,
) -> bool {
    if let (Some(lower), Some(upper)) = (inputs[0].dense(), inputs[1].dense()) {
        let (lower_out, upper_out) = (outputs[0], outputs[1]);
        // Inputs address `len` lanes, and outputs are fresh `len`-element tensors.
        return if same {
            run_lanes(client, len, |i| unsafe {
                let span = Span {
                    lo: dense_get::<T>(lower, i),
                    hi: dense_get::<T>(upper, i),
                };
                lower_out.put(i, span.lo);
                span.invalid_by_compare()
            })
        } else {
            run_lanes(client, len, |i| unsafe {
                let span = Span {
                    lo: dense_get::<T>(lower, i),
                    hi: dense_get::<T>(upper, i),
                };
                lower_out.put(i, span.lo);
                upper_out.put(i, span.hi);
                span.invalid_by_compare()
            })
        };
    }
    run_lanes(client, len, |i| {
        // Inputs address `len` lanes, and outputs are fresh `len`-element tensors.
        unsafe {
            let span = Span {
                lo: inputs[0].get::<T>(i),
                hi: inputs[1].get::<T>(i),
            };
            outputs[0].put(i, span.lo);
            if !same {
                outputs[1].put(i, span.hi);
            }
            span.invalid()
        }
    })
}

pub(super) fn sqrt_lanes<T: LaneFloat>(
    client: &CpuClient,
    inputs: [Input; 2],
    outputs: [Output; 2],
    len: usize,
) -> bool {
    run_lanes(client, len, |i| {
        // Inputs address `len` lanes, and outputs are fresh `len`-element tensors.
        unsafe {
            let (out, flag) = sqrt_nonnegative(Span {
                lo: inputs[0].get::<T>(i),
                hi: inputs[1].get::<T>(i),
            });
            outputs[0].put(i, out.lo);
            outputs[1].put(i, out.hi);
            flag
        }
    })
}
