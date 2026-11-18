//! CUDA `CorrectlyRoundedOps` against the CPU tensor ops, bit for bit.

use numr::prelude::*;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

use super::inputs::{HYDR_REGRESSION, adversarial, dense, exponents};
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::create_cpu_client;

/// One tensor op of the trait.
#[derive(Debug, Clone, Copy)]
enum Op {
    Exp,
    Log,
    Log1p,
    Pow(f64),
}

fn ops() -> Vec<Op> {
    let mut ops = vec![Op::Exp, Op::Log, Op::Log1p];
    ops.extend(exponents().into_iter().map(Op::Pow));
    ops
}

fn apply<R, C>(client: &C, op: Op, a: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: CorrectlyRoundedOps<R>,
{
    match op {
        Op::Exp => client.cr_exp(a),
        Op::Log => client.cr_log(a),
        Op::Log1p => client.cr_log1p(a),
        Op::Pow(y) => client.cr_pow_scalar(a, y),
    }
}

/// Panic with the first differing elements when the bit patterns differ.
fn assert_same_bits(op: Op, label: &str, inputs: &[f64], cpu: &[f64], cuda: &[f64]) {
    assert_eq!(
        cpu.len(),
        cuda.len(),
        "{op:?} {label}: output lengths differ"
    );
    let mismatches: Vec<String> = inputs
        .iter()
        .zip(cpu.iter().zip(cuda))
        .filter(|(_, (c, g))| c.to_bits() != g.to_bits())
        .map(|(x, (c, g))| {
            format!(
                "x={:#018x} cpu={:#018x} cuda={:#018x}",
                x.to_bits(),
                c.to_bits(),
                g.to_bits()
            )
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "{op:?} {label}: {} of {} elements differ, first: {:?}",
        mismatches.len(),
        inputs.len(),
        &mismatches[..mismatches.len().min(8)]
    );
}

/// Run `op` on `data` as a 1-D tensor on both runtimes and compare every bit.
fn check(client: &CudaClient, device: &CudaDevice, op: Op, data: &[f64], label: &str) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_in = Tensor::<CpuRuntime>::from_slice(data, &[data.len()], &cpu_device).unwrap();
    let cuda_in = Tensor::<CudaRuntime>::from_slice(data, &[data.len()], device).unwrap();
    let cpu = apply(&cpu_client, op, &cpu_in).unwrap().to_vec::<f64>();
    let cuda = apply(client, op, &cuda_in).unwrap().to_vec::<f64>();
    assert_same_bits(op, label, data, &cpu, &cuda);
}

#[test]
fn correctly_rounded_cuda_matches_cpu_on_adversarial_inputs() {
    with_cuda_backend(|client, device| {
        let data = adversarial();
        for op in ops() {
            check(&client, &device, op, &data, "adversarial");
        }
    });
}

#[test]
fn correctly_rounded_cuda_matches_cpu_on_dense_bit_patterns() {
    with_cuda_backend(|client, device| {
        let data = dense(1 << 17, 0x5eed_c0de);
        for op in ops() {
            check(&client, &device, op, &data, "dense");
        }
    });
}

#[test]
fn correctly_rounded_cuda_matches_cpu_on_hydr_regression_input() {
    with_cuda_backend(|client, device| {
        for op in ops() {
            check(&client, &device, op, &[HYDR_REGRESSION], "hydr regression");
        }
    });
}

#[test]
fn correctly_rounded_cuda_matches_cpu_across_block_boundaries() {
    with_cuda_backend(|client, device| {
        let data = dense(40_000, 0xb10c);
        for len in [
            1usize, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4097, 65_537,
            80_000,
        ] {
            for op in [Op::Exp, Op::Log, Op::Log1p, Op::Pow(4.0 / 3.0)] {
                check(&client, &device, op, &data[..len], &format!("length {len}"));
            }
        }
    });
}

#[test]
fn correctly_rounded_cuda_matches_cpu_on_strided_views() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        let (rows, cols) = (48usize, 70usize);
        let data = dense(rows * cols / 2, 0x57_71de);
        let cpu_base = Tensor::<CpuRuntime>::from_slice(&data, &[rows, cols], &cpu_device).unwrap();
        let cuda_base = Tensor::<CudaRuntime>::from_slice(&data, &[rows, cols], &device).unwrap();
        type View<R> = fn(&Tensor<R>) -> Tensor<R>;
        let views: [(&str, View<CpuRuntime>, View<CudaRuntime>); 5] = [
            (
                "transpose",
                |t| t.transpose(0, 1).unwrap(),
                |t| t.transpose(0, 1).unwrap(),
            ),
            (
                "column slice",
                |t| t.narrow(1, 3, 41).unwrap(),
                |t| t.narrow(1, 3, 41).unwrap(),
            ),
            (
                "row slice",
                |t| t.narrow(0, 5, 17).unwrap(),
                |t| t.narrow(0, 5, 17).unwrap(),
            ),
            (
                "transposed slice",
                |t| t.narrow(1, 7, 9).unwrap().transpose(0, 1).unwrap(),
                |t| t.narrow(1, 7, 9).unwrap().transpose(0, 1).unwrap(),
            ),
            (
                "one element",
                |t| t.narrow(0, 9, 1).unwrap().narrow(1, 4, 1).unwrap(),
                |t| t.narrow(0, 9, 1).unwrap().narrow(1, 4, 1).unwrap(),
            ),
        ];
        for (label, cpu_view, cuda_view) in views {
            let cpu_in = cpu_view(&cpu_base);
            let cuda_in = cuda_view(&cuda_base);
            for op in [
                Op::Exp,
                Op::Log,
                Op::Log1p,
                Op::Pow(7.0 / 3.0),
                Op::Pow(-1.0),
            ] {
                let cpu_out = apply(&cpu_client, op, &cpu_in).unwrap();
                let cuda_out = apply(&client, op, &cuda_in).unwrap();
                assert_eq!(cuda_out.shape(), cpu_in.shape(), "{op:?} {label}: shape");
                assert!(cuda_out.is_contiguous(), "{op:?} {label}: output layout");
                let inputs = cpu_in.contiguous().unwrap().to_vec::<f64>();
                assert_same_bits(
                    op,
                    label,
                    &inputs,
                    &cpu_out.to_vec::<f64>(),
                    &cuda_out.to_vec::<f64>(),
                );
            }
        }
    });
}

#[test]
fn correctly_rounded_cuda_handles_empty_scalar_and_rejected_dtypes() {
    with_cuda_backend(|client, device| {
        let empty = Tensor::<CudaRuntime>::empty(&[0, 3], DType::F64, &device).unwrap();
        let scalar = Tensor::<CudaRuntime>::from_slice(&[0.5f64], &[], &device).unwrap();
        for op in ops() {
            let out = apply(&client, op, &empty).unwrap();
            assert_eq!(out.shape(), &[0, 3]);
            let out = apply(&client, op, &scalar).unwrap();
            assert_eq!(out.shape(), &[] as &[usize]);
        }
        check(&client, &device, Op::Exp, &[0.5], "one element");
        let names = ["cr_exp", "cr_log", "cr_log1p", "cr_pow_scalar"];
        for dtype in [DType::F32, DType::I32, DType::I64, DType::U32] {
            let input = Tensor::<CudaRuntime>::empty(&[4], dtype, &device).unwrap();
            for (op, name) in [Op::Exp, Op::Log, Op::Log1p, Op::Pow(2.0)]
                .into_iter()
                .zip(names)
            {
                match apply(&client, op, &input) {
                    Err(Error::UnsupportedDType {
                        dtype: got,
                        op: got_op,
                    }) => {
                        assert_eq!((got, got_op), (dtype, name));
                    }
                    other => panic!("{name} {dtype:?}: {other:?}"),
                }
            }
        }
    });
}
