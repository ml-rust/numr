//! Determinism check for WebGPU `scatter_reduce`.
//!
//! Colliding contributions to one destination element must combine in one
//! order on every run. A float atomic per source element lands in scheduling
//! order, and float addition is not associative, so such a kernel returns
//! different low bits from one run to the next while staying inside any
//! parity tolerance.
//!
//! WebGPU reduces each destination in increasing source position, the order
//! the CPU reference uses. Each case therefore asserts two things on raw bits:
//! every WebGPU run matches the first, and the first matches CPU.
//!
//! Run: cargo test --release --features wgpu --test wgpu_scatter_reduce_determinism

#![cfg(feature = "wgpu")]

mod common;

use common::backend_lock::with_wgpu_backend_or_skip;
use common::create_cpu_client;
use numr::ops::{IndexingOps, ScatterReduceOp};
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
use numr::tensor::Tensor;

/// Runs per case. A scheduling-order race need not fire on every launch.
const REPEATS: usize = 20;

const OPS: [ScatterReduceOp; 5] = [
    ScatterReduceOp::Sum,
    ScatterReduceOp::Prod,
    ScatterReduceOp::Max,
    ScatterReduceOp::Min,
    ScatterReduceOp::Mean,
];

/// A 32-bit integer hash of `i`, in `[0, 1)`.
fn unit(i: u64) -> f64 {
    let mut x = i.wrapping_mul(2_654_435_761) & 0xFFFF_FFFF;
    x ^= x >> 16;
    x = x.wrapping_mul(0x045D_9F3B) & 0xFFFF_FFFF;
    x ^= x >> 16;
    x as f64 / (1u64 << 32) as f64
}

/// One scatter_reduce call, as plain data both backends can stage.
struct Case<T> {
    dst: Vec<T>,
    dst_shape: Vec<usize>,
    src: Vec<T>,
    index: Vec<i64>,
    src_shape: Vec<usize>,
    dim: usize,
}

trait Bits: Copy + numr::dtype::Element + bytemuck::Pod {
    fn bits(self) -> u64;
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Bits for i32 {
    fn bits(self) -> u64 {
        u64::from(self as u32)
    }
}

impl Bits for u32 {
    fn bits(self) -> u64 {
        u64::from(self)
    }
}

/// The index tensor's element type on the WebGPU side.
#[derive(Debug, Clone, Copy)]
enum IndexType {
    I32,
    I64,
}

fn run_cpu<T: Bits>(case: &Case<T>, op: ScatterReduceOp, include_self: bool) -> Vec<u64> {
    let (client, device) = create_cpu_client();
    let dst = Tensor::<CpuRuntime>::from_slice(&case.dst, &case.dst_shape, &device)
        .expect("staging a tensor must succeed");
    let src = Tensor::<CpuRuntime>::from_slice(&case.src, &case.src_shape, &device)
        .expect("staging a tensor must succeed");
    let index = Tensor::<CpuRuntime>::from_slice(&case.index, &case.src_shape, &device)
        .expect("staging a tensor must succeed");
    client
        .scatter_reduce(&dst, case.dim, &index, &src, op, include_self)
        .expect("CPU scatter_reduce must succeed")
        .to_vec::<T>()
        .into_iter()
        .map(Bits::bits)
        .collect()
}

fn run_wgpu<T: Bits>(
    client: &WgpuClient,
    device: &WgpuDevice,
    case: &Case<T>,
    index_type: IndexType,
    op: ScatterReduceOp,
    include_self: bool,
) -> Vec<u64> {
    // Fresh buffers each run: a reused output would only show the kernel is
    // idempotent on memory it already wrote.
    let dst = Tensor::<WgpuRuntime>::from_slice(&case.dst, &case.dst_shape, device)
        .expect("staging a tensor must succeed");
    let src = Tensor::<WgpuRuntime>::from_slice(&case.src, &case.src_shape, device)
        .expect("staging a tensor must succeed");
    let index = match index_type {
        IndexType::I64 => Tensor::<WgpuRuntime>::from_slice(&case.index, &case.src_shape, device),
        IndexType::I32 => {
            let narrow: Vec<i32> = case.index.iter().map(|&i| i as i32).collect();
            Tensor::<WgpuRuntime>::from_slice(&narrow, &case.src_shape, device)
        }
    }
    .expect("staging a tensor must succeed");
    client
        .scatter_reduce(&dst, case.dim, &index, &src, op, include_self)
        .expect("WebGPU scatter_reduce must succeed")
        .to_vec::<T>()
        .into_iter()
        .map(Bits::bits)
        .collect()
}

/// Panics unless every WebGPU run and the CPU reference give the same bits.
#[allow(clippy::too_many_arguments)]
fn assert_deterministic<T: Bits>(
    client: &WgpuClient,
    device: &WgpuDevice,
    label: &str,
    case: &Case<T>,
    index_type: IndexType,
    op: ScatterReduceOp,
    include_self: bool,
) {
    let want = run_cpu(case, op, include_self);
    for run in 0..REPEATS {
        let got = run_wgpu(client, device, case, index_type, op, include_self);
        assert_eq!(got.len(), want.len(), "{label} {op:?}: length");
        if let Some(at) = got.iter().zip(&want).position(|(g, w)| g != w) {
            panic!(
                "{label} {op:?} include_self={include_self} index={index_type:?}: run {run} \
                 element {at} has bits {:#x} on WebGPU and {:#x} on CPU",
                got[at], want[at]
            );
        }
    }
}

/// Overlap-add as an inverse STFT runs it: frame `t` sample `k` lands at
/// `t * hop + k`, so each output sample sums `n_fft / hop` frames.
fn overlap_add(frames: usize, n_fft: usize, hop: usize) -> Case<f32> {
    let n = frames * n_fft;
    let out_len = (frames - 1) * hop + n_fft;
    Case {
        dst: vec![0.0; out_len],
        dst_shape: vec![1, out_len],
        src: (0..n)
            .map(|i| (unit(i as u64) * 2.0 - 1.0) as f32)
            .collect(),
        index: (0..n)
            .map(|i| ((i / n_fft) * hop + i % n_fft) as i64)
            .collect(),
        src_shape: vec![1, n],
        dim: 1,
    }
}

/// A `[3, 7, 4]` destination scattered on axis 1 from a `[3, 40, 2]` source:
/// the source is shorter on the last axis, and some indices fall outside
/// `[0, 7)`. Values sit near 1 so a product stays finite.
fn mixed_case<T: Bits>(cast: impl Fn(f64) -> T) -> Case<T> {
    let dst_shape = vec![3, 7, 4];
    let src_shape = vec![3, 40, 2];
    let dst_n: usize = dst_shape.iter().product();
    let src_n: usize = src_shape.iter().product();
    Case {
        dst: (0..dst_n)
            .map(|i| cast(0.5 + unit(i as u64 + 7_000)))
            .collect(),
        dst_shape,
        src: (0..src_n).map(|i| cast(0.5 + unit(i as u64))).collect(),
        index: (0..src_n)
            .map(|i| (unit(i as u64 + 1_000) * 9.0) as i64 - 1)
            .collect(),
        src_shape,
        dim: 1,
    }
}

#[test]
fn overlap_add_sum_is_bit_identical_run_to_run() {
    with_wgpu_backend_or_skip(|client, device| {
        let case = overlap_add(2000, 20, 5);
        assert_deterministic(
            &client,
            &device,
            "overlap-add",
            &case,
            IndexType::I64,
            ScatterReduceOp::Sum,
            true,
        );
    });
}

#[test]
fn every_f32_op_matches_cpu_bits() {
    with_wgpu_backend_or_skip(|client, device| {
        let case = mixed_case(|v| v as f32);
        for op in OPS {
            for include_self in [false, true] {
                for index_type in [IndexType::I64, IndexType::I32] {
                    assert_deterministic(
                        &client,
                        &device,
                        "f32",
                        &case,
                        index_type,
                        op,
                        include_self,
                    );
                }
            }
        }
    });
}

#[test]
fn integer_ops_match_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let signed = mixed_case(|v| ((v - 0.9) * 1000.0) as i32);
        let unsigned = mixed_case(|v| (v * 1000.0) as u32);
        for op in OPS {
            for include_self in [false, true] {
                assert_deterministic(
                    &client,
                    &device,
                    "i32",
                    &signed,
                    IndexType::I64,
                    op,
                    include_self,
                );
                assert_deterministic(
                    &client,
                    &device,
                    "u32",
                    &unsigned,
                    IndexType::I32,
                    op,
                    include_self,
                );
            }
        }
    });
}

/// `mean` divides without float division. Destination `d` takes `1 + d % 37`
/// contributions spread over many binades, so the quotients land on every
/// rounding case the f64 round trip on CPU produces.
#[test]
fn mean_division_rounds_like_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let dst_n = 4096;
        let mut index = Vec::new();
        for d in 0..dst_n {
            index.extend(std::iter::repeat_n(d as i64, 1 + d % 37));
        }
        // Interleave destinations so each run gathers from far-apart sources.
        index.sort_by_key(|&d| (unit(d as u64 * 31) * 1e9) as i64);
        let src_n = index.len();
        let src: Vec<f32> = (0..src_n)
            .map(|i| {
                let scale = 2f64.powi((unit(i as u64 + 5) * 120.0) as i32 - 60);
                ((unit(i as u64) - 0.3) * scale) as f32
            })
            .collect();
        let case = Case {
            dst: (0..dst_n)
                .map(|i| (unit(i as u64 + 77) * 3.0) as f32)
                .collect(),
            dst_shape: vec![dst_n],
            src,
            index,
            src_shape: vec![src_n],
            dim: 0,
        };
        for include_self in [false, true] {
            assert_deterministic(
                &client,
                &device,
                "mean",
                &case,
                IndexType::I64,
                ScatterReduceOp::Mean,
                include_self,
            );
        }
    });
}

/// A destination past 2^16 elements needs three radix passes, and a source
/// past one radix tile spreads every digit across tiles.
#[test]
fn many_tiles_and_three_passes_match_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let dst_n = 200_000;
        let src_n = 300_000;
        let case = Case {
            dst: vec![0.0f32; dst_n],
            dst_shape: vec![dst_n],
            src: (0..src_n).map(|i| (unit(i as u64) - 0.5) as f32).collect(),
            index: (0..src_n)
                .map(|i| (unit(i as u64 + 99) * dst_n as f64) as i64)
                .collect(),
            src_shape: vec![src_n],
            dim: 0,
        };
        assert_deterministic(
            &client,
            &device,
            "large",
            &case,
            IndexType::I64,
            ScatterReduceOp::Sum,
            true,
        );
        assert_deterministic(
            &client,
            &device,
            "large",
            &case,
            IndexType::I32,
            ScatterReduceOp::Mean,
            false,
        );
    });
}

#[test]
fn a_source_longer_off_the_scatter_axis_is_rejected() {
    with_wgpu_backend_or_skip(|client, device| {
        let dst = Tensor::<WgpuRuntime>::from_slice(&[0.0f32; 6], &[2, 3], &device)
            .expect("staging a tensor must succeed");
        let src = Tensor::<WgpuRuntime>::from_slice(&[1.0f32; 9], &[3, 3], &device)
            .expect("staging a tensor must succeed");
        let index = Tensor::<WgpuRuntime>::from_slice(&[0i64; 9], &[3, 3], &device)
            .expect("staging a tensor must succeed");
        let err = client
            .scatter_reduce(&dst, 1, &index, &src, ScatterReduceOp::Sum, true)
            .expect_err("a source longer than the destination on axis 0 must be refused");
        assert!(err.to_string().contains("axis 0"), "{err}");
    });
}

#[test]
fn an_empty_source_leaves_the_seed() {
    with_wgpu_backend_or_skip(|client, device| {
        let dst = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, -0.0, 3.0], &[3], &device)
            .expect("staging a tensor must succeed");
        let src = Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0], &device)
            .expect("staging a tensor must succeed");
        let index = Tensor::<WgpuRuntime>::from_slice::<i64>(&[], &[0], &device)
            .expect("staging a tensor must succeed");
        for _ in 0..REPEATS {
            let kept = client
                .scatter_reduce(&dst, 0, &index, &src, ScatterReduceOp::Sum, true)
                .expect("an empty source scatters nothing");
            let bits: Vec<u32> = kept.to_vec::<f32>().iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                bits,
                vec![1.0f32.to_bits(), (-0.0f32).to_bits(), 3.0f32.to_bits()]
            );

            let identity = client
                .scatter_reduce(&dst, 0, &index, &src, ScatterReduceOp::Max, false)
                .expect("an empty source scatters nothing");
            assert_eq!(identity.to_vec::<f32>(), vec![f32::NEG_INFINITY; 3]);
        }
    });
}
