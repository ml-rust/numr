//! CPU certified expansion operations through single-pass lane kernels.

use super::interval_lanes::{cpu_operand, input};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::impl_generic::interval::{
    ExpansionBinary, LaneOperand, check_expansions, lane_count, require_basic, require_fused,
    validate,
};
use crate::ops::{ExpansionOperand, ExpansionOps, ExpansionTriple, IntervalOperand, IntervalPair};
use crate::runtime::cpu::kernels::interval::drive::{Input, Output, dense_get, raise, run_lanes};
use crate::runtime::cpu::kernels::interval::lane_division;
use crate::runtime::cpu::kernels::interval::lane_expansion::{
    self, FlaggedParts, Parts, normalize, project,
};
use crate::runtime::cpu::kernels::interval::lane_float::LaneFloat;
use crate::runtime::cpu::kernels::interval::lane_interval::Span;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

type CpuTensor = Tensor<CpuRuntime>;

/// Lane inputs of one expansion operand.
fn parts_inputs<'a>(
    value: ExpansionOperand<'a, CpuRuntime>,
) -> Result<([Input; 3], [LaneOperand<'a, CpuRuntime>; 3])> {
    let shape = value.high.shape();
    let operands = [
        cpu_operand(value.high, shape)?,
        cpu_operand(value.low, shape)?,
        cpu_operand(value.abs_error, shape)?,
    ];
    let inputs = operands.each_ref().map(input);
    Ok((inputs, operands))
}

/// # Safety
///
/// Each input addresses more than `index` lanes.
#[inline(always)]
unsafe fn read<T: LaneFloat>(inputs: [Input; 3], index: usize) -> Parts<T> {
    unsafe {
        Parts {
            high: inputs[0].get(index),
            low: inputs[1].get(index),
            error: inputs[2].get(index),
        }
    }
}

fn outputs(client: &CpuClient, shape: &[usize], dtype: DType) -> Result<[CpuTensor; 3]> {
    Ok([
        CpuTensor::empty(shape, dtype, &client.device)?,
        CpuTensor::empty(shape, dtype, &client.device)?,
        CpuTensor::empty(shape, dtype, &client.device)?,
    ])
}

/// Run a lane rule over expansion operands and write three components.
fn run_parts<T, F>(
    client: &CpuClient,
    sources: [[Input; 3]; 2],
    out: &[CpuTensor; 3],
    rule: F,
) -> bool
where
    T: LaneFloat,
    F: Fn(Parts<T>, Parts<T>) -> FlaggedParts<T> + Send + Sync,
{
    let targets = out.each_ref().map(Output::of);
    let dense = |source: [Input; 3]| -> Option<[usize; 3]> {
        Some([source[0].dense()?, source[1].dense()?, source[2].dense()?])
    };
    if let (Some(a), Some(b)) = (dense(sources[0]), dense(sources[1])) {
        let read_dense = |addr: [usize; 3], i: usize| unsafe {
            Parts {
                high: dense_get::<T>(addr[0], i),
                low: dense_get::<T>(addr[1], i),
                error: dense_get::<T>(addr[2], i),
            }
        };
        return run_lanes(client, out[0].numel(), move |i| {
            // Inputs address every lane, and outputs are fresh tensors of the lane count.
            let (value, flag) = rule(read_dense(a, i), read_dense(b, i));
            unsafe {
                targets[0].put(i, value.high);
                targets[1].put(i, value.low);
                targets[2].put(i, value.error);
            }
            flag
        });
    }
    run_lanes(client, out[0].numel(), |i| {
        // Inputs address every lane, and outputs are fresh tensors of the lane count.
        unsafe {
            let (value, flag) = rule(read(sources[0], i), read(sources[1], i));
            targets[0].put(i, value.high);
            targets[1].put(i, value.low);
            targets[2].put(i, value.error);
            flag
        }
    })
}

/// Run a one-operand lane rule. Generic rules inline into the lane loop.
fn unary_parts<F32, F64>(
    client: &CpuClient,
    value: ExpansionOperand<'_, CpuRuntime>,
    status: &CpuTensor,
    rule32: F32,
    rule64: F64,
    op: &'static str,
) -> Result<ExpansionTriple<CpuRuntime>>
where
    F32: Fn(Parts<f32>) -> FlaggedParts<f32> + Send + Sync,
    F64: Fn(Parts<f64>) -> FlaggedParts<f64> + Send + Sync,
{
    let dtype = check_expansions(client, op, &[value], status)?;
    let out = outputs(client, value.high.shape(), dtype)?;
    if out[0].numel() > 0 {
        let (inputs, _keep) = parts_inputs(value)?;
        let flagged = match dtype {
            DType::F32 => run_parts(client, [inputs; 2], &out, |x: Parts<f32>, _| rule32(x)),
            _ => run_parts(client, [inputs; 2], &out, |x: Parts<f64>, _| rule64(x)),
        };
        raise(status, flagged);
    }
    let [high, low, abs_error] = out;
    Ok((high, low, abs_error))
}

fn binary_parts(
    client: &CpuClient,
    op: ExpansionBinary,
    a: ExpansionOperand<'_, CpuRuntime>,
    b: ExpansionOperand<'_, CpuRuntime>,
    status: &CpuTensor,
) -> Result<ExpansionTriple<CpuRuntime>> {
    let dtype = check_expansions(client, op.name(), &[a, b], status)?;
    if op.needs_fused() {
        require_fused::<CpuRuntime>(dtype, op.name())?;
    }
    let out = outputs(client, a.high.shape(), dtype)?;
    if out[0].numel() > 0 {
        let (a_inputs, _keep_a) = parts_inputs(a)?;
        let (b_inputs, _keep_b) = parts_inputs(b)?;
        let sources = [a_inputs, b_inputs];
        macro_rules! run {
            ($t:ty) => {
                match op {
                    ExpansionBinary::Add => run_parts::<$t, _>(client, sources, &out, |x, y| {
                        lane_expansion::sum(x, y, false)
                    }),
                    ExpansionBinary::Sub => run_parts::<$t, _>(client, sources, &out, |x, y| {
                        lane_expansion::sum(x, y, true)
                    }),
                    ExpansionBinary::Mul => {
                        run_parts::<$t, _>(client, sources, &out, lane_expansion::product)
                    }
                    ExpansionBinary::DivPositive => {
                        run_parts::<$t, _>(client, sources, &out, lane_division::quotient)
                    }
                }
            };
        }
        let flagged = match dtype {
            DType::F32 => run!(f32),
            _ => run!(f64),
        };
        raise(status, flagged);
    }
    let [high, low, abs_error] = out;
    Ok((high, low, abs_error))
}

fn point_rule<T: LaneFloat>(value: Parts<T>) -> FlaggedParts<T> {
    let out = Parts {
        high: value.high,
        low: T::ZERO,
        error: T::ZERO,
    };
    (out, !value.high.finite())
}

fn flag_rule<T: LaneFloat>(value: Parts<T>) -> FlaggedParts<T> {
    (value, value.invalid())
}

impl ExpansionOps<CpuRuntime> for CpuClient {
    fn expansion_point(
        &self,
        value: &CpuTensor,
        status: &CpuTensor,
    ) -> Result<ExpansionTriple<CpuRuntime>> {
        let view = ExpansionOperand {
            high: value,
            low: value,
            abs_error: value,
        };
        unary_parts(
            self,
            view,
            status,
            point_rule,
            point_rule,
            "expansion_point",
        )
    }

    fn expansion_normalize(
        &self,
        value: ExpansionOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<ExpansionTriple<CpuRuntime>> {
        unary_parts(
            self,
            value,
            status,
            normalize,
            normalize,
            "expansion_normalize",
        )
    }

    fn expansion_flag(
        &self,
        value: ExpansionOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<()> {
        let dtype = check_expansions(self, "expansion_flag", &[value], status)?;
        let len = value.high.numel();
        if len == 0 {
            return Ok(());
        }
        let (inputs, _keep) = parts_inputs(value)?;
        let flagged = match dtype {
            DType::F32 => run_lanes(self, len, |i| unsafe {
                flag_rule(read::<f32>(inputs, i)).1
            }),
            _ => run_lanes(self, len, |i| unsafe {
                flag_rule(read::<f64>(inputs, i)).1
            }),
        };
        raise(status, flagged);
        Ok(())
    }

    fn expansion_add(
        &self,
        a: ExpansionOperand<'_, CpuRuntime>,
        b: ExpansionOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<ExpansionTriple<CpuRuntime>> {
        binary_parts(self, ExpansionBinary::Add, a, b, status)
    }

    fn expansion_sub(
        &self,
        a: ExpansionOperand<'_, CpuRuntime>,
        b: ExpansionOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<ExpansionTriple<CpuRuntime>> {
        binary_parts(self, ExpansionBinary::Sub, a, b, status)
    }

    fn expansion_mul(
        &self,
        a: ExpansionOperand<'_, CpuRuntime>,
        b: ExpansionOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<ExpansionTriple<CpuRuntime>> {
        binary_parts(self, ExpansionBinary::Mul, a, b, status)
    }

    fn expansion_div_positive(
        &self,
        a: ExpansionOperand<'_, CpuRuntime>,
        b: ExpansionOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<ExpansionTriple<CpuRuntime>> {
        binary_parts(self, ExpansionBinary::DivPositive, a, b, status)
    }

    fn expansion_from_interval(
        &self,
        value: IntervalOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<ExpansionTriple<CpuRuntime>> {
        let dtype =
            validate::float_dtype(self, "expansion_from_interval", &[value.lower, value.upper])?;
        validate::interval(value)?;
        validate::status(self, status)?;
        require_basic::<CpuRuntime>(dtype, "expansion_from_interval")?;
        let shape = value.lower.shape();
        let out = outputs(self, shape, dtype)?;
        if out[0].numel() > 0 {
            let operands = [
                cpu_operand(value.lower, shape)?,
                cpu_operand(value.upper, shape)?,
            ];
            let [lower, upper] = operands.each_ref().map(input);
            let targets = out.each_ref().map(Output::of);
            macro_rules! run {
                ($t:ty) => {
                    run_lanes(self, out[0].numel(), |i| {
                        // Inputs address every lane, and outputs are fresh tensors of the lane count.
                        unsafe {
                            let (value, flag) = lane_division::lift(Span::<$t> {
                                lo: lower.get(i),
                                hi: upper.get(i),
                            });
                            targets[0].put(i, value.high);
                            targets[1].put(i, value.low);
                            targets[2].put(i, value.error);
                            flag
                        }
                    })
                };
            }
            let flagged = match dtype {
                DType::F32 => run!(f32),
                _ => run!(f64),
            };
            raise(status, flagged);
        }
        let [high, low, abs_error] = out;
        Ok((high, low, abs_error))
    }

    fn expansion_interval(
        &self,
        value: ExpansionOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<IntervalPair<CpuRuntime>> {
        let dtype = check_expansions(self, "expansion_interval", &[value], status)?;
        let shape = value.high.shape();
        let lower = CpuTensor::empty(shape, dtype, &self.device)?;
        let upper = CpuTensor::empty(shape, dtype, &self.device)?;
        let len = lane_count(shape);
        if len > 0 {
            let (inputs, _keep) = parts_inputs(value)?;
            let targets = [Output::of(&lower), Output::of(&upper)];
            macro_rules! run {
                ($t:ty) => {
                    run_lanes(self, len, |i| {
                        // Inputs address every lane, and outputs are fresh tensors of the lane count.
                        unsafe {
                            let (span, flag) = project(read::<$t>(inputs, i));
                            targets[0].put(i, span.lo);
                            targets[1].put(i, span.hi);
                            flag
                        }
                    })
                };
            }
            let flagged = match dtype {
                DType::F32 => run!(f32),
                _ => run!(f64),
            };
            raise(status, flagged);
        }
        Ok((lower, upper))
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::impl_generic::interval::parity_expansion::expansion_ops;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn expansion_kernels_match_composed_bits_and_flags() -> crate::error::Result<()> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        expansion_ops::<CpuRuntime, _, f64>(&client, &device, 20_000, 300)?;
        expansion_ops::<CpuRuntime, _, f32>(&client, &device, 20_000, 300)
    }
}
