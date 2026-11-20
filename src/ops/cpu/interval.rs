//! CPU interval operations through single-pass lane kernels.

use super::interval_lanes::{
    capture_lanes, cpu_operand, flag_interval, flag_spans, input, sqrt_lanes,
};
use crate::dispatch_dtype;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::impl_generic::interval::envelope::EnvelopeConstants;
use crate::ops::impl_generic::interval::{
    IntervalBinary, binary_shape, check_binary, interval_status_impl, lane_count, validate,
};
use crate::ops::{IntervalEnclosure, IntervalOperand, IntervalOps, IntervalPair};
use crate::runtime::cpu::kernels::interval::drive::{Input, Output, dense_get, raise, run_lanes};
use crate::runtime::cpu::kernels::interval::envelope::probe;
use crate::runtime::cpu::kernels::interval::lane_float::LaneFloat;
use crate::runtime::cpu::kernels::interval::lane_interval::Span;
use crate::runtime::cpu::kernels::interval::lane_public::{self, LaneMode};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::runtime::{Device, FusedMulAddContract, Runtime};
use crate::tensor::Tensor;

type CpuTensor = Tensor<CpuRuntime>;

fn run_binary<T: LaneFloat>(
    client: &CpuClient,
    op: IntervalBinary,
    mode: LaneMode,
    inputs: [Input; 4],
    outputs: [Output; 2],
    len: usize,
) -> bool {
    // Each rule and mode pair gets its own inlined lane loop.
    macro_rules! modes {
        ($rule:expr) => {{
            let rule = $rule;
            match (mode.tight, mode.fused) {
                (true, true) => run_rule(client, inputs, outputs, len, rule, TIGHT_FUSED),
                (true, false) => run_rule(client, inputs, outputs, len, rule, TIGHT_UNFUSED),
                (false, true) => run_rule(client, inputs, outputs, len, rule, OUTWARD_FUSED),
                (false, false) => run_rule(client, inputs, outputs, len, rule, OUTWARD_UNFUSED),
            }
        }};
    }
    match op {
        IntervalBinary::Add => modes!(lane_public::add::<T>),
        IntervalBinary::Sub => modes!(lane_public::sub::<T>),
        IntervalBinary::Mul => modes!(lane_public::mul::<T>),
        IntervalBinary::DivPositive => {
            modes!(|a: Span<T>, b: Span<T>, mode| lane_public::div_positive(a, b.lo, mode))
        }
        IntervalBinary::DivPositiveInterval => modes!(lane_public::div_positive_interval::<T>),
    }
}

const TIGHT_FUSED: LaneMode = LaneMode {
    tight: true,
    fused: true,
};
const TIGHT_UNFUSED: LaneMode = LaneMode {
    tight: true,
    fused: false,
};
const OUTWARD_FUSED: LaneMode = LaneMode {
    tight: false,
    fused: true,
};
const OUTWARD_UNFUSED: LaneMode = LaneMode {
    tight: false,
    fused: false,
};

#[inline(always)]
fn run_rule<T, F>(
    client: &CpuClient,
    inputs: [Input; 4],
    outputs: [Output; 2],
    len: usize,
    rule: F,
    mode: LaneMode,
) -> bool
where
    T: LaneFloat,
    F: Fn(Span<T>, Span<T>, LaneMode) -> (Span<T>, bool) + Send + Sync,
{
    let [a_lo, a_hi, b_lo, b_hi] = inputs;
    let [out_lo, out_hi] = outputs;
    if let (Some(al), Some(ah), Some(bl), Some(bh)) =
        (a_lo.dense(), a_hi.dense(), b_lo.dense(), b_hi.dense())
    {
        return run_lanes(client, len, move |i| {
            // Operands address `len` lanes, and outputs are fresh `len`-element tensors.
            unsafe {
                let a = Span {
                    lo: dense_get::<T>(al, i),
                    hi: dense_get::<T>(ah, i),
                };
                let b = Span {
                    lo: dense_get::<T>(bl, i),
                    hi: dense_get::<T>(bh, i),
                };
                let (out, flag) = rule(a, b, mode);
                out_lo.put(i, out.lo);
                out_hi.put(i, out.hi);
                flag
            }
        });
    }
    run_lanes(client, len, move |i| {
        // Operands address `len` lanes, and outputs are fresh `len`-element tensors.
        unsafe {
            let a = Span {
                lo: a_lo.get::<T>(i),
                hi: a_hi.get::<T>(i),
            };
            let b = Span {
                lo: b_lo.get::<T>(i),
                hi: b_hi.get::<T>(i),
            };
            let (out, flag) = rule(a, b, mode);
            out_lo.put(i, out.lo);
            out_hi.put(i, out.hi);
            flag
        }
    })
}

fn binary(
    client: &CpuClient,
    op: IntervalBinary,
    a: IntervalOperand<'_, CpuRuntime>,
    b: IntervalOperand<'_, CpuRuntime>,
    enclosure: IntervalEnclosure,
    status: &CpuTensor,
) -> Result<IntervalPair<CpuRuntime>> {
    let dtype = check_binary(client, op.name(), a, b, status)?;
    let shape = match binary_shape(op, a, b) {
        Ok(shape) => shape,
        Err(error) => {
            input_flags(client, op, dtype, a, b, status)?;
            return Err(error);
        }
    };
    let lower = CpuTensor::empty(&shape, dtype, &client.device)?;
    let upper = CpuTensor::empty(&shape, dtype, &client.device)?;
    let len = lane_count(&shape);
    if len == 0 {
        input_flags(client, op, dtype, a, b, status)?;
        return Ok((lower, upper));
    }
    let operands = [
        cpu_operand(a.lower, &shape)?,
        cpu_operand(a.upper, &shape)?,
        cpu_operand(b.lower, &shape)?,
        cpu_operand(b.upper, &shape)?,
    ];
    let inputs = operands.each_ref().map(input);
    let outputs = [Output::of(&lower), Output::of(&upper)];
    let mode = LaneMode {
        tight: enclosure == IntervalEnclosure::Tightened,
        fused: CpuRuntime::fused_mul_add_contract(dtype) == FusedMulAddContract::Ieee754Fused,
    };
    let flagged = match dtype {
        DType::F32 => run_binary::<f32>(client, op, mode, inputs, outputs, len),
        _ => run_binary::<f64>(client, op, mode, inputs, outputs, len),
    };
    raise(status, flagged);
    Ok((lower, upper))
}

/// Mark operand-shaped division checks that no output lane covers.
fn input_flags(
    client: &CpuClient,
    op: IntervalBinary,
    dtype: DType,
    a: IntervalOperand<'_, CpuRuntime>,
    b: IntervalOperand<'_, CpuRuntime>,
    status: &CpuTensor,
) -> Result<()> {
    match op {
        IntervalBinary::DivPositive => flag_interval(client, dtype, b, true, status),
        IntervalBinary::DivPositiveInterval => {
            flag_interval(client, dtype, a, false, status)?;
            flag_interval(client, dtype, b, true, status)
        }
        _ => Ok(()),
    }
}

impl IntervalOps<CpuRuntime> for CpuClient {
    fn interval_status(&self) -> Result<CpuTensor> {
        interval_status_impl(self)
    }

    fn interval_envelope(&self, dtype: DType) -> Result<CpuTensor> {
        let constants = EnvelopeConstants::for_dtype(dtype)?;
        let bits = match dtype {
            DType::F32 => probe::<f32>(constants),
            _ => probe::<f64>(constants),
        };
        CpuTensor::from_slice(&[bits], &[], &self.device)
    }

    fn interval_flag_nonzero(&self, mask: &CpuTensor, status: &CpuTensor) -> Result<()> {
        validate::status(self, status)?;
        if !self.device.is_same(mask.device()) {
            return Err(Error::DeviceMismatch);
        }
        if mask.numel() == 0 {
            return Ok(());
        }
        let operand = cpu_operand(mask, mask.shape())?;
        let source = input(&operand);
        let flagged = dispatch_dtype!(mask.dtype(), T => {
            run_lanes(self, mask.numel(), |i| {
                // The operand addresses every lane of `mask`.
                unsafe { source.get::<T>(i) != T::zero() }
            })
        }, "interval_flag_nonzero");
        raise(status, flagged);
        Ok(())
    }

    fn interval_flag_nonfinite(&self, value: &CpuTensor, status: &CpuTensor) -> Result<()> {
        let dtype = validate::float_dtype(self, "interval_flag_nonfinite", &[value])?;
        validate::status(self, status)?;
        if value.numel() == 0 {
            return Ok(());
        }
        let operand = cpu_operand(value, value.shape())?;
        let source = input(&operand);
        let flagged = match dtype {
            DType::F32 => flag_spans::<f32>(self, source, source, value.numel(), false),
            _ => flag_spans::<f64>(self, source, source, value.numel(), false),
        };
        raise(status, flagged);
        Ok(())
    }

    fn interval_flag_bounds(
        &self,
        value: IntervalOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<()> {
        let dtype =
            validate::float_dtype(self, "interval_flag_bounds", &[value.lower, value.upper])?;
        validate::interval(value)?;
        validate::status(self, status)?;
        flag_interval(self, dtype, value, false, status)
    }

    fn interval_capture(
        &self,
        value: IntervalOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<IntervalPair<CpuRuntime>> {
        let dtype = validate::float_dtype(self, "interval_capture", &[value.lower, value.upper])?;
        validate::interval(value)?;
        validate::status(self, status)?;
        let shape = value.lower.shape();
        let same = validate::same_view(value.lower, value.upper);
        let lower = CpuTensor::empty(shape, dtype, &self.device)?;
        let upper = if same {
            lower.clone()
        } else {
            CpuTensor::empty(shape, dtype, &self.device)?
        };
        let len = lane_count(shape);
        if len > 0 {
            let operands = [
                cpu_operand(value.lower, shape)?,
                cpu_operand(value.upper, shape)?,
            ];
            let inputs = operands.each_ref().map(input);
            let outputs = [Output::of(&lower), Output::of(&upper)];
            let flagged = match dtype {
                DType::F32 => capture_lanes::<f32>(self, inputs, outputs, len, same),
                _ => capture_lanes::<f64>(self, inputs, outputs, len, same),
            };
            raise(status, flagged);
        }
        Ok((lower, upper))
    }

    fn interval_add(
        &self,
        a: IntervalOperand<'_, CpuRuntime>,
        b: IntervalOperand<'_, CpuRuntime>,
        enclosure: IntervalEnclosure,
        status: &CpuTensor,
    ) -> Result<IntervalPair<CpuRuntime>> {
        binary(self, IntervalBinary::Add, a, b, enclosure, status)
    }

    fn interval_sub(
        &self,
        a: IntervalOperand<'_, CpuRuntime>,
        b: IntervalOperand<'_, CpuRuntime>,
        enclosure: IntervalEnclosure,
        status: &CpuTensor,
    ) -> Result<IntervalPair<CpuRuntime>> {
        binary(self, IntervalBinary::Sub, a, b, enclosure, status)
    }

    fn interval_mul(
        &self,
        a: IntervalOperand<'_, CpuRuntime>,
        b: IntervalOperand<'_, CpuRuntime>,
        enclosure: IntervalEnclosure,
        status: &CpuTensor,
    ) -> Result<IntervalPair<CpuRuntime>> {
        binary(self, IntervalBinary::Mul, a, b, enclosure, status)
    }

    fn interval_div_positive(
        &self,
        a: IntervalOperand<'_, CpuRuntime>,
        divisor: &CpuTensor,
        enclosure: IntervalEnclosure,
        status: &CpuTensor,
    ) -> Result<IntervalPair<CpuRuntime>> {
        let b = IntervalOperand {
            lower: divisor,
            upper: divisor,
        };
        binary(self, IntervalBinary::DivPositive, a, b, enclosure, status)
    }

    fn interval_div_positive_interval(
        &self,
        a: IntervalOperand<'_, CpuRuntime>,
        b: IntervalOperand<'_, CpuRuntime>,
        enclosure: IntervalEnclosure,
        status: &CpuTensor,
    ) -> Result<IntervalPair<CpuRuntime>> {
        binary(
            self,
            IntervalBinary::DivPositiveInterval,
            a,
            b,
            enclosure,
            status,
        )
    }

    fn interval_sqrt_nonnegative(
        &self,
        a: IntervalOperand<'_, CpuRuntime>,
        status: &CpuTensor,
    ) -> Result<IntervalPair<CpuRuntime>> {
        let dtype = validate::float_dtype(self, "interval_sqrt_nonnegative", &[a.lower, a.upper])?;
        validate::interval(a)?;
        validate::status(self, status)?;
        let shape = a.lower.shape();
        let lower = CpuTensor::empty(shape, dtype, &self.device)?;
        let upper = CpuTensor::empty(shape, dtype, &self.device)?;
        let len = lane_count(shape);
        if len == 0 {
            return Ok((lower, upper));
        }
        let operands = [cpu_operand(a.lower, shape)?, cpu_operand(a.upper, shape)?];
        let inputs = operands.each_ref().map(input);
        let outputs = [Output::of(&lower), Output::of(&upper)];
        let flagged = match dtype {
            DType::F32 => sqrt_lanes::<f32>(self, inputs, outputs, len),
            _ => sqrt_lanes::<f64>(self, inputs, outputs, len),
        };
        raise(status, flagged);
        Ok((lower, upper))
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::impl_generic::interval::parity_interval::{
        binary_edges, binary_flags, binary_grid, unary_ops,
    };
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn interval_kernels_match_composed_bits() -> crate::error::Result<()> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        binary_grid::<CpuRuntime, _, f64>(&client, &device)?;
        binary_grid::<CpuRuntime, _, f32>(&client, &device)
    }

    #[test]
    fn interval_kernels_match_composed_lane_flags() -> crate::error::Result<()> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        binary_flags::<CpuRuntime, _, f64>(&client, &device, 7)?;
        binary_flags::<CpuRuntime, _, f32>(&client, &device, 11)?;
        binary_edges::<CpuRuntime, _, f64>(&client, &device)?;
        binary_edges::<CpuRuntime, _, f32>(&client, &device)
    }

    #[test]
    fn envelope_kernel_matches_composed_probes() -> crate::error::Result<()> {
        use crate::dtype::DType;
        use crate::ops::IntervalOps;
        use crate::ops::impl_generic::interval::envelope::interval_envelope_impl;
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        for dtype in [DType::F32, DType::F64] {
            let native = client.interval_envelope(dtype)?.item::<u32>()?;
            let reference = interval_envelope_impl(&client, dtype)?.item::<u32>()?;
            assert_eq!(native, reference, "{dtype:?}");
        }
        Ok(())
    }

    #[test]
    fn interval_unary_kernels_match_composed_bits_and_flags() -> crate::error::Result<()> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        unary_ops::<CpuRuntime, _, f64>(&client, &device)?;
        unary_ops::<CpuRuntime, _, f32>(&client, &device)
    }
}
