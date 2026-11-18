//! `CorrectlyRoundedOps` on CPU tensors against the host functions, bit for bit.
//!
//! The host functions match MPFR in the sibling modules, so tensor bits equal to host bits
//! are correctly rounded too.

use super::adversarial::specials;
use super::hard_cases::load;
use super::oracle::Function;
use super::pow_hard_cases::load_pow;
use super::pow_special::hydr_exponents;
use numr::algorithm::correctly_rounded as cr;
use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;
use numr::runtime::cpu::correctly_rounded_lanes::{CrFunction, LaneCopy, map_f64};

/// One tensor op and its host function.
#[derive(Debug, Clone, Copy)]
enum Op {
    Unary(Function),
    Pow(f64),
}

impl Op {
    fn host(self, x: f64) -> f64 {
        match self {
            Op::Unary(function) => function.numr(x),
            Op::Pow(y) => cr::pow(x, y),
        }
    }

    fn tensor(self, client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        match self {
            Op::Unary(Function::Exp) => client.cr_exp(a),
            Op::Unary(Function::Log) => client.cr_log(a),
            Op::Unary(Function::Log1p) => client.cr_log1p(a),
            Op::Pow(y) => client.cr_pow_scalar(a, y),
        }
    }

    fn lanes(self) -> CrFunction {
        match self {
            Op::Unary(Function::Exp) => CrFunction::Exp,
            Op::Unary(Function::Log) => CrFunction::Log,
            Op::Unary(Function::Log1p) => CrFunction::Log1p,
            Op::Pow(y) => CrFunction::PowScalar(y),
        }
    }
}

fn ops() -> Vec<Op> {
    let mut ops: Vec<Op> = Function::ALL.into_iter().map(Op::Unary).collect();
    ops.extend(hydr_exponents().into_iter().map(Op::Pow));
    ops
}

/// Clients that split work differently: one thread, and four threads with odd chunks.
fn clients() -> Vec<(CpuClient, &'static str)> {
    let base = CpuClient::new(CpuDevice::new());
    vec![
        (
            base.with_parallelism(ParallelismConfig::new(Some(1), None)),
            "1 thread",
        ),
        (
            base.with_parallelism(ParallelismConfig::new(Some(4), Some(37))),
            "4 threads, chunk 37",
        ),
        (
            base.with_parallelism(ParallelismConfig::new(Some(3), Some(1))),
            "3 threads, chunk 1",
        ),
        (base, "default"),
    ]
}

/// Adversarial values first, then the hard cases of `op`, one in `stride`.
fn pool(op: Op, stride: usize) -> Vec<f64> {
    let mut v = specials();
    match op {
        Op::Unary(function) => v.extend(load(function).into_iter().step_by(stride)),
        Op::Pow(_) => v.extend(load_pow().into_iter().step_by(stride).map(|(x, _)| x)),
    }
    v
}

/// Assert `actual` holds `op` of `input`, element by element, bit for bit.
fn check_bits(op: Op, input: &[f64], actual: &[f64], label: &str) {
    assert_eq!(input.len(), actual.len(), "{op:?} {label}: length");
    for (i, (&x, &y)) in input.iter().zip(actual).enumerate() {
        let want = op.host(x);
        assert_eq!(
            y.to_bits(),
            want.to_bits(),
            "{op:?} {label}: lane {i}, x={:#018x}",
            x.to_bits()
        );
    }
}

fn run(op: Op, client: &CpuClient, input: &Tensor<CpuRuntime>, label: &str) -> Result<()> {
    let expected_input: Vec<f64> = input.contiguous()?.try_to_vec()?;
    let out = op.tensor(client, input)?;
    assert_eq!(out.shape(), input.shape(), "{op:?} {label}: shape");
    assert!(out.is_contiguous(), "{op:?} {label}: layout");
    check_bits(op, &expected_input, &out.try_to_vec::<f64>()?, label);
    Ok(())
}

#[test]
fn test_tensor_equals_host_across_lengths_and_thread_counts() -> Result<()> {
    let lengths: Vec<usize> = (0..=100)
        .chain([255, 256, 257, 4095, 4096, 4097, 50_000])
        .collect();
    let device = CpuDevice::new();
    for op in ops() {
        let pool = pool(op, 97);
        for (client, name) in clients() {
            for &len in &lengths {
                let data: Vec<f64> = pool.iter().copied().cycle().take(len).collect();
                let input = Tensor::<CpuRuntime>::from_slice(&data, &[len], &device)?;
                run(op, &client, &input, &format!("{name}, len {len}"))?;
            }
        }
    }
    Ok(())
}

#[test]
fn test_tensor_equals_host_on_strided_views() -> Result<()> {
    let device = CpuDevice::new();
    let (rows, cols) = (33, 65);
    for op in ops() {
        let data: Vec<f64> = pool(op, 211)
            .into_iter()
            .cycle()
            .take(rows * cols)
            .collect();
        let base = Tensor::<CpuRuntime>::from_slice(&data, &[rows, cols], &device)?;
        let views = [
            (base.transpose(0, 1)?, "transposed"),
            (base.narrow(1, 3, 31)?, "narrowed columns"),
            (base.narrow(0, 1, 32)?.flip(1)?, "offset and reversed"),
            (
                base.transpose(0, 1)?.narrow(0, 5, 40)?,
                "transposed and narrowed",
            ),
        ];
        for (client, name) in clients() {
            for (view, kind) in &views {
                run(op, &client, view, &format!("{name}, {kind}"))?;
            }
        }
    }
    Ok(())
}

#[test]
fn test_tensor_equals_host_on_every_hard_case() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone())
        .with_parallelism(ParallelismConfig::new(Some(4), Some(1000)));
    for function in Function::ALL {
        let op = Op::Unary(function);
        let data = load(function);
        let input = Tensor::<CpuRuntime>::from_slice(&data, &[data.len()], &device)?;
        run(op, &client, &input, "hard cases")?;
    }
    Ok(())
}

#[test]
fn test_pow_tensor_equals_host_on_hard_case_groups() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let pairs = load_pow();
    // The fixture lists pairs grouped by exponent; each group is one tensor call.
    let same_y = |a: &(f64, f64), b: &(f64, f64)| a.1.to_bits() == b.1.to_bits();
    let mut groups = 0usize;
    for group in pairs.chunk_by(same_y) {
        groups += 1;
        let y = group[0].1;
        let data: Vec<f64> = group.iter().map(|&(x, _)| x).collect();
        let input = Tensor::<CpuRuntime>::from_slice(&data, &[data.len()], &device)?;
        run(Op::Pow(y), &client, &input, "pow hard cases")?;
    }
    assert!(groups > 8, "pow fixture holds {groups} exponents");
    Ok(())
}

#[test]
fn test_fused_and_plain_lane_copies_agree_on_hard_cases() {
    let mut cases: Vec<(Op, Vec<f64>)> = Function::ALL
        .into_iter()
        .map(|function| (Op::Unary(function), load(function)))
        .collect();
    for group in load_pow().chunk_by(|a, b| a.1.to_bits() == b.1.to_bits()) {
        cases.push((Op::Pow(group[0].1), group.iter().map(|&(x, _)| x).collect()));
    }
    for (op, input) in cases {
        let mut fused = vec![0.0; input.len()];
        let mut plain = vec![0.0; input.len()];
        map_f64(op.lanes(), &input, &mut fused, LaneCopy::Best);
        map_f64(op.lanes(), &input, &mut plain, LaneCopy::Plain);
        check_bits(op, &input, &fused, "fused copy");
        check_bits(op, &input, &plain, "plain copy");
    }
}
