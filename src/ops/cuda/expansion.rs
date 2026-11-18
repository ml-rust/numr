//! CUDA certified expansion operations through single-pass kernels.

use super::interval_lanes::{cuda_operand, lanes, launch, operand_args, suffix};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::impl_generic::interval::{
    ExpansionBinary, check_expansions, require_basic, require_fused, validate,
};
use crate::ops::{ExpansionOperand, ExpansionOps, ExpansionTriple, IntervalOperand, IntervalPair};
use crate::runtime::cuda::kernels::IntervalArg;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

type CudaTensor = Tensor<CudaRuntime>;

/// Kernel operation codes, matching `interval.cu`.
#[derive(Clone, Copy)]
enum Code {
    Add = 0,
    Sub = 1,
    Mul = 2,
    DivPositive = 3,
    Point = 4,
    Normalize = 5,
    Flag = 6,
    Project = 7,
    Lift = 8,
}

/// Fresh component tensors shaped like `like`.
fn fresh<const N: usize>(
    client: &CudaClient,
    like: &CudaTensor,
    dtype: DType,
) -> Result<[CudaTensor; N]> {
    let mut out = Vec::with_capacity(N);
    for _ in 0..N {
        out.push(CudaTensor::empty(like.shape(), dtype, &client.device)?);
    }
    out.try_into()
        .map_err(|_| crate::error::Error::Internal("expansion output count".into()))
}

/// Launch one expansion kernel over identical-shape operands into fresh `out` tensors.
///
/// `out` holds three components, two projection endpoints, or nothing for flags.
#[allow(clippy::too_many_arguments)]
fn run(
    client: &CudaClient,
    code: Code,
    name: &'static str,
    dtype: DType,
    a: [&CudaTensor; 3],
    b: [&CudaTensor; 3],
    out: &[CudaTensor],
    status: &CudaTensor,
) -> Result<()> {
    let shape = a[0].shape();
    let count = a[0].numel();
    if count == 0 {
        return Ok(());
    }
    let n = lanes(count, name)?;
    let operands = a
        .iter()
        .chain(&b)
        .map(|value| cuda_operand(client, value, shape))
        .collect::<Result<Vec<_>>>()?;
    let mut args = vec![IntervalArg::I32(code as i32)];
    args.extend(operands.iter().flat_map(operand_args));
    // Unused output slots repeat the status address; those kernels never write them.
    for k in 0..3 {
        args.push(IntervalArg::Ptr(
            out.get(k).map_or(status.ptr(), CudaTensor::ptr),
        ));
    }
    args.extend([IntervalArg::Ptr(status.ptr()), IntervalArg::U32(n)]);
    launch(
        client,
        &format!("expansion_{}", suffix(dtype)),
        &args,
        count,
    )
}

fn components<'a>(value: ExpansionOperand<'a, CudaRuntime>) -> [&'a CudaTensor; 3] {
    [value.high, value.low, value.abs_error]
}

fn triple(out: [CudaTensor; 3]) -> ExpansionTriple<CudaRuntime> {
    let [high, low, abs_error] = out;
    (high, low, abs_error)
}

fn binary(
    client: &CudaClient,
    op: ExpansionBinary,
    a: ExpansionOperand<'_, CudaRuntime>,
    b: ExpansionOperand<'_, CudaRuntime>,
    status: &CudaTensor,
) -> Result<ExpansionTriple<CudaRuntime>> {
    let dtype = check_expansions(client, op.name(), &[a, b], status)?;
    if op.needs_fused() {
        require_fused::<CudaRuntime>(dtype, op.name())?;
    }
    let code = match op {
        ExpansionBinary::Add => Code::Add,
        ExpansionBinary::Sub => Code::Sub,
        ExpansionBinary::Mul => Code::Mul,
        ExpansionBinary::DivPositive => Code::DivPositive,
    };
    let out = fresh::<3>(client, a.high, dtype)?;
    run(
        client,
        code,
        op.name(),
        dtype,
        components(a),
        components(b),
        &out,
        status,
    )?;
    Ok(triple(out))
}

fn unary<const N: usize>(
    client: &CudaClient,
    code: Code,
    name: &'static str,
    value: ExpansionOperand<'_, CudaRuntime>,
    status: &CudaTensor,
) -> Result<[CudaTensor; N]> {
    let dtype = check_expansions(client, name, &[value], status)?;
    let out = fresh::<N>(client, value.high, dtype)?;
    let parts = components(value);
    run(client, code, name, dtype, parts, parts, &out, status)?;
    Ok(out)
}

impl ExpansionOps<CudaRuntime> for CudaClient {
    fn expansion_point(
        &self,
        value: &CudaTensor,
        status: &CudaTensor,
    ) -> Result<ExpansionTriple<CudaRuntime>> {
        let view = ExpansionOperand {
            high: value,
            low: value,
            abs_error: value,
        };
        unary::<3>(self, Code::Point, "expansion_point", view, status).map(triple)
    }

    fn expansion_normalize(
        &self,
        value: ExpansionOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<ExpansionTriple<CudaRuntime>> {
        unary::<3>(self, Code::Normalize, "expansion_normalize", value, status).map(triple)
    }

    fn expansion_flag(
        &self,
        value: ExpansionOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<()> {
        unary::<0>(self, Code::Flag, "expansion_flag", value, status).map(|_| ())
    }

    fn expansion_add(
        &self,
        a: ExpansionOperand<'_, CudaRuntime>,
        b: ExpansionOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<ExpansionTriple<CudaRuntime>> {
        binary(self, ExpansionBinary::Add, a, b, status)
    }

    fn expansion_sub(
        &self,
        a: ExpansionOperand<'_, CudaRuntime>,
        b: ExpansionOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<ExpansionTriple<CudaRuntime>> {
        binary(self, ExpansionBinary::Sub, a, b, status)
    }

    fn expansion_mul(
        &self,
        a: ExpansionOperand<'_, CudaRuntime>,
        b: ExpansionOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<ExpansionTriple<CudaRuntime>> {
        binary(self, ExpansionBinary::Mul, a, b, status)
    }

    fn expansion_div_positive(
        &self,
        a: ExpansionOperand<'_, CudaRuntime>,
        b: ExpansionOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<ExpansionTriple<CudaRuntime>> {
        binary(self, ExpansionBinary::DivPositive, a, b, status)
    }

    fn expansion_from_interval(
        &self,
        value: IntervalOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<ExpansionTriple<CudaRuntime>> {
        let dtype =
            validate::float_dtype(self, "expansion_from_interval", &[value.lower, value.upper])?;
        validate::interval(value)?;
        validate::status(self, status)?;
        require_basic::<CudaRuntime>(dtype, "expansion_from_interval")?;
        let parts = [value.lower, value.upper, value.lower];
        let out = fresh::<3>(self, value.lower, dtype)?;
        run(
            self,
            Code::Lift,
            "expansion_from_interval",
            dtype,
            parts,
            parts,
            &out,
            status,
        )?;
        Ok(triple(out))
    }

    fn expansion_interval(
        &self,
        value: ExpansionOperand<'_, CudaRuntime>,
        status: &CudaTensor,
    ) -> Result<IntervalPair<CudaRuntime>> {
        let [lower, upper] = unary::<2>(self, Code::Project, "expansion_interval", value, status)?;
        Ok((lower, upper))
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::impl_generic::interval::parity_expansion::expansion_ops;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::{CudaDevice, CudaRuntime};

    #[test]
    fn cuda_expansion_kernels_match_composed_bits_and_flags() -> crate::error::Result<()> {
        if !crate::runtime::cuda::is_cuda_available() {
            return Ok(());
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        expansion_ops::<CudaRuntime, _, f64>(&client, &device, 20_000, 60)?;
        expansion_ops::<CudaRuntime, _, f32>(&client, &device, 20_000, 60)
    }
}
