//! CUDA interval operations through single-pass kernels.

use super::interval_lanes::{
    cuda_operand, flag_interval, input_flags, lanes, launch, operand_args, suffix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::impl_generic::interval::envelope::EnvelopeConstants;
use crate::ops::impl_generic::interval::{
    IntervalBinary, binary_shape, check_binary, interval_status_impl, lane_count, validate,
};
use crate::ops::{IntervalEnclosure, IntervalOperand, IntervalOps, IntervalPair};
use crate::runtime::cuda::kernels::IntervalArg;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{Device, FusedMulAddContract, Runtime};
use crate::tensor::Tensor;

type CudaTensor = Tensor<CudaRuntime>;

fn code(op: IntervalBinary) -> i32 {
    match op {
        IntervalBinary::Add => 0,
        IntervalBinary::Sub => 1,
        IntervalBinary::Mul => 2,
        IntervalBinary::DivPositive => 3,
        IntervalBinary::DivPositiveInterval => 4,
    }
}

fn binary(
    client: &CudaClient,
    op: IntervalBinary,
    a: IntervalOperand<'_, CudaRuntime>,
    b: IntervalOperand<'_, CudaRuntime>,
    enclosure: IntervalEnclosure,
    status: &CudaTensor,
) -> Result<IntervalPair<CudaRuntime>> {
    let dtype = check_binary(client, op.name(), a, b, status)?;
    let shape = match binary_shape(op, a, b) {
        Ok(shape) => shape,
        Err(error) => {
            input_flags(client, op, dtype, a, b, status)?;
            return Err(error);
        }
    };
    let lower = CudaTensor::empty(&shape, dtype, &client.device)?;
    let upper = CudaTensor::empty(&shape, dtype, &client.device)?;
    let count = lane_count(&shape);
    if count == 0 {
        input_flags(client, op, dtype, a, b, status)?;
        return Ok((lower, upper));
    }
    let n = lanes(count, op.name())?;
    let operands = [
        cuda_operand(client, a.lower, &shape)?,
        cuda_operand(client, a.upper, &shape)?,
        cuda_operand(client, b.lower, &shape)?,
        cuda_operand(client, b.upper, &shape)?,
    ];
    let fused = CudaRuntime::fused_mul_add_contract(dtype) == FusedMulAddContract::Ieee754Fused;
    let mut args = vec![
        IntervalArg::I32(code(op)),
        IntervalArg::I32(i32::from(enclosure == IntervalEnclosure::Tightened)),
        IntervalArg::I32(i32::from(fused)),
    ];
    args.extend(operands.iter().flat_map(operand_args));
    args.extend([
        IntervalArg::Ptr(lower.ptr()),
        IntervalArg::Ptr(upper.ptr()),
        IntervalArg::Ptr(status.ptr()),
        IntervalArg::U32(n),
    ]);
    launch(
        client,
        &format!("interval_binary_{}", suffix(dtype)),
        &args,
        count,
    )?;
    Ok((lower, upper))
}

/// Run a two-input, two-output kernel: square roots or captures.
fn unary(
    client: &CudaClient,
    kernel: &'static str,
    name: &'static str,
    value: IntervalOperand<'_, CudaRuntime>,
    same: Option<bool>,
    status: &CudaTensor,
) -> Result<IntervalPair<CudaRuntime>> {
    let dtype = validate::float_dtype(client, name, &[value.lower, value.upper])?;
    validate::interval(value)?;
    validate::status(client, status)?;
    let shape = value.lower.shape();
    let lower = CudaTensor::empty(shape, dtype, &client.device)?;
    let upper = if same == Some(true) {
        lower.clone()
    } else {
        CudaTensor::empty(shape, dtype, &client.device)?
    };
    let count = lane_count(shape);
    if count == 0 {
        return Ok((lower, upper));
    }
    let n = lanes(count, name)?;
    let operands = [
        cuda_operand(client, value.lower, shape)?,
        cuda_operand(client, value.upper, shape)?,
    ];
    let mut args: Vec<IntervalArg> = operands.iter().flat_map(operand_args).collect();
    args.extend([IntervalArg::Ptr(lower.ptr()), IntervalArg::Ptr(upper.ptr())]);
    if let Some(same) = same {
        args.push(IntervalArg::I32(i32::from(same)));
    }
    args.extend([IntervalArg::Ptr(status.ptr()), IntervalArg::U32(n)]);
    launch(client, &format!("{kernel}_{}", suffix(dtype)), &args, count)?;
    Ok((lower, upper))
}

fn mask_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::I64 => Ok("i64"),
        DType::I32 => Ok("i32"),
        DType::U32 => Ok("u32"),
        DType::U8 | DType::Bool => Ok("u8"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "interval_flag_nonzero",
        }),
    }
}

impl IntervalOps<CudaRuntime> for CudaClient {
    fn interval_status(&self) -> Result<CudaTensor> {
        interval_status_impl(self)
    }

    fn interval_envelope(&self, dtype: DType) -> Result<CudaTensor> {
        let k = EnvelopeConstants::for_dtype(dtype)?;
        let out = CudaTensor::empty(&[], DType::U32, &self.device)?;
        let values = [
            k.epsilon,
            k.minimum_subnormal,
            k.minimum_normal,
            1.0 + 2.0 * k.epsilon,
            2.0 * k.epsilon,
        ];
        let mut args: Vec<IntervalArg> = values
            .iter()
            .map(|&value| match dtype {
                DType::F32 => IntervalArg::F32(value as f32),
                _ => IntervalArg::F64(value),
            })
            .collect();
        args.push(IntervalArg::Ptr(out.ptr()));
        launch(
            self,
            &format!("interval_envelope_{}", suffix(dtype)),
            &args,
            1,
        )?;
        Ok(out)
    }

    fn interval_flag_nonzero(&self, mask: &CudaTensor, status: &CudaTensor) -> Result<()> {
        validate::status(self, status)?;
        if !self.device.is_same(mask.device()) {
            return Err(Error::DeviceMismatch);
        }
        let name = mask_suffix(mask.dtype())?;
        let count = mask.numel();
        if count == 0 {
            return Ok(());
        }
        let n = lanes(count, "interval_flag_nonzero")?;
        let operand = cuda_operand(self, mask, mask.shape())?;
        let [ptr, step] = operand_args(&operand);
        let args = [
            ptr,
            step,
            IntervalArg::Ptr(status.ptr()),
            IntervalArg::U32(n),
        ];
        launch(self, &format!("interval_flag_nonzero_{name}"), &args, count)
    }

    fn interval_flag_nonfinite(&self, value: &CudaTensor, status: &CudaTensor) -> Result<()> {
        let dtype = validate::float_dtype(self, "interval_flag_nonfinite", &[value])?;
        validate::status(self, status)?;
        let view = IntervalOperand {
            lower: value,
            upper: value,
        };
        flag_interval(self, dtype, view, false, status)
    }

    fn interval_flag_bounds(
        &self,
        value: IntervalOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<()> {
        let dtype =
            validate::float_dtype(self, "interval_flag_bounds", &[value.lower, value.upper])?;
        validate::interval(value)?;
        validate::status(self, status)?;
        flag_interval(self, dtype, value, false, status)
    }

    fn interval_capture(
        &self,
        value: IntervalOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<IntervalPair<CudaRuntime>> {
        let same = validate::same_view(value.lower, value.upper);
        unary(
            self,
            "interval_capture",
            "interval_capture",
            value,
            Some(same),
            status,
        )
    }

    fn interval_add(
        &self,
        a: IntervalOperand<'_, CudaRuntime>,
        b: IntervalOperand<'_, CudaRuntime>,
        enclosure: IntervalEnclosure,
        status: &CudaTensor,
    ) -> Result<IntervalPair<CudaRuntime>> {
        binary(self, IntervalBinary::Add, a, b, enclosure, status)
    }

    fn interval_sub(
        &self,
        a: IntervalOperand<'_, CudaRuntime>,
        b: IntervalOperand<'_, CudaRuntime>,
        enclosure: IntervalEnclosure,
        status: &CudaTensor,
    ) -> Result<IntervalPair<CudaRuntime>> {
        binary(self, IntervalBinary::Sub, a, b, enclosure, status)
    }

    fn interval_mul(
        &self,
        a: IntervalOperand<'_, CudaRuntime>,
        b: IntervalOperand<'_, CudaRuntime>,
        enclosure: IntervalEnclosure,
        status: &CudaTensor,
    ) -> Result<IntervalPair<CudaRuntime>> {
        binary(self, IntervalBinary::Mul, a, b, enclosure, status)
    }

    fn interval_div_positive(
        &self,
        a: IntervalOperand<'_, CudaRuntime>,
        divisor: &CudaTensor,
        enclosure: IntervalEnclosure,
        status: &CudaTensor,
    ) -> Result<IntervalPair<CudaRuntime>> {
        let b = IntervalOperand {
            lower: divisor,
            upper: divisor,
        };
        binary(self, IntervalBinary::DivPositive, a, b, enclosure, status)
    }

    fn interval_div_positive_interval(
        &self,
        a: IntervalOperand<'_, CudaRuntime>,
        b: IntervalOperand<'_, CudaRuntime>,
        enclosure: IntervalEnclosure,
        status: &CudaTensor,
    ) -> Result<IntervalPair<CudaRuntime>> {
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
        a: IntervalOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<IntervalPair<CudaRuntime>> {
        unary(
            self,
            "interval_sqrt",
            "interval_sqrt_nonnegative",
            a,
            None,
            status,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::IntervalOps;
    use crate::ops::impl_generic::interval::envelope::interval_envelope_impl;
    use crate::ops::impl_generic::interval::parity_interval::{
        binary_edges, binary_flags, binary_grid, unary_ops,
    };
    use crate::runtime::Runtime;
    use crate::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        let device = CudaDevice::new(0);
        Some((CudaRuntime::default_client(&device), device))
    }

    #[test]
    fn cuda_interval_kernels_match_composed_bits() -> crate::error::Result<()> {
        let Some((client, device)) = setup() else {
            return Ok(());
        };
        binary_grid::<CudaRuntime, _, f64>(&client, &device)?;
        binary_grid::<CudaRuntime, _, f32>(&client, &device)
    }

    #[test]
    fn cuda_interval_kernels_match_composed_lane_flags() -> crate::error::Result<()> {
        let Some((client, device)) = setup() else {
            return Ok(());
        };
        binary_flags::<CudaRuntime, _, f64>(&client, &device, 19)?;
        binary_flags::<CudaRuntime, _, f32>(&client, &device, 23)?;
        binary_edges::<CudaRuntime, _, f64>(&client, &device)?;
        binary_edges::<CudaRuntime, _, f32>(&client, &device)
    }

    #[test]
    fn cuda_interval_unary_and_envelope_match_composed() -> crate::error::Result<()> {
        let Some((client, device)) = setup() else {
            return Ok(());
        };
        unary_ops::<CudaRuntime, _, f64>(&client, &device)?;
        unary_ops::<CudaRuntime, _, f32>(&client, &device)?;
        for dtype in [crate::dtype::DType::F32, crate::dtype::DType::F64] {
            let native = client.interval_envelope(dtype)?.item::<u32>()?;
            let reference = interval_envelope_impl(&client, dtype)?.item::<u32>()?;
            assert_eq!(native, reference, "{dtype:?}");
        }
        Ok(())
    }
}
