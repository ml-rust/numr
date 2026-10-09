//! CPU cdist and pdist output must not depend on the thread pool.
//!
//! cdist splits its output into column blocks of one row, and pdist into rows.
//! The split is a pure function of the shape. These tests run each op on
//! clients with different pool sizes and chunk sizes and compare raw bits.
//! The shapes sit above the parallel work threshold, so the pooled runs fork.
//!
//! F16 and BF16 run as F32 inside the op. Their tests also check the output
//! against an f64 reference, within a bound derived from `d` and the format.

use numr::dtype::Element;
use numr::ops::{DistanceMetric, DistanceOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

const METRICS: [DistanceMetric; 5] = [
    DistanceMetric::SquaredEuclidean,
    DistanceMetric::Euclidean,
    DistanceMetric::Cosine,
    DistanceMetric::Manhattan,
    DistanceMetric::Chebyshev,
];

/// Clients that schedule the same units differently.
fn clients(device: &CpuDevice) -> Vec<(&'static str, CpuClient)> {
    let base = CpuClient::new(device.clone());
    vec![
        (
            "1 thread",
            base.with_parallelism(ParallelismConfig::new(Some(1), None)),
        ),
        (
            "4 threads",
            base.with_parallelism(ParallelismConfig::new(Some(4), None)),
        ),
        ("default", base.clone()),
        (
            "4 threads, chunk 7",
            base.with_parallelism(ParallelismConfig::new(Some(4), Some(7))),
        ),
    ]
}

fn values(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 251) as f64) * 0.004 - 0.5)
        .collect()
}

/// Element types whose output bits the tests compare.
trait Bits: Element {
    fn bits(self) -> u64;
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Bits for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

#[cfg(feature = "f16")]
impl Bits for half::f16 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

#[cfg(feature = "f16")]
impl Bits for half::bf16 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

fn tensor<T: Element>(data: &[f64], shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let typed: Vec<T> = data.iter().map(|&v| T::from_f64(v)).collect();
    Tensor::<CpuRuntime>::from_slice(&typed, shape, device).unwrap()
}

/// Asserts that every client writes the bits of the first, and returns the
/// first client's output as f64.
fn assert_schedule_invariant<T: Bits>(
    label: &str,
    run: impl Fn(&CpuClient) -> Tensor<CpuRuntime>,
    clients: &[(&str, CpuClient)],
) -> Vec<f64> {
    let reference: Vec<T> = run(&clients[0].1).to_vec();
    for (name, client) in &clients[1..] {
        let got: Vec<T> = run(client).to_vec();
        assert_eq!(got.len(), reference.len(), "{label}: length on {name}");
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert_eq!(
                g.bits(),
                r.bits(),
                "{label}: element {i} differs on {name} ({} vs {})",
                g.to_f64(),
                r.to_f64()
            );
        }
    }
    reference.iter().map(|v| v.to_f64()).collect()
}

fn cdist_case<T: Bits>(n: usize, m: usize, d: usize) -> Vec<(DistanceMetric, Vec<f64>)> {
    let device = CpuDevice::new();
    let x = tensor::<T>(&values(n * d, 1), &[n, d], &device);
    let y = tensor::<T>(&values(m * d, 2), &[m, d], &device);
    let clients = clients(&device);
    METRICS
        .iter()
        .map(|&metric| {
            let label = format!("cdist {metric:?} {:?} n={n} m={m} d={d}", T::DTYPE);
            let out = assert_schedule_invariant::<T>(
                &label,
                |c| c.cdist(&x, &y, metric).unwrap(),
                &clients,
            );
            (metric, out)
        })
        .collect()
}

fn pdist_case<T: Bits>(n: usize, d: usize) -> Vec<(DistanceMetric, Vec<f64>)> {
    let device = CpuDevice::new();
    let x = tensor::<T>(&values(n * d, 3), &[n, d], &device);
    let clients = clients(&device);
    METRICS
        .iter()
        .map(|&metric| {
            let label = format!("pdist {metric:?} {:?} n={n} d={d}", T::DTYPE);
            let run = |c: &CpuClient| c.pdist(&x, metric).unwrap();
            let out = assert_schedule_invariant::<T>(&label, run, &clients);
            (metric, out)
        })
        .collect()
}

#[test]
fn cdist_f32_is_schedule_invariant() {
    cdist_case::<f32>(48, 130, 257);
}

#[test]
fn cdist_f64_is_schedule_invariant() {
    cdist_case::<f64>(48, 130, 257);
}

/// One query against many rows: the units are column blocks of a single row.
#[test]
fn cdist_single_query_is_schedule_invariant() {
    cdist_case::<f32>(1, 500, 257);
    cdist_case::<f64>(1, 500, 257);
}

#[test]
fn pdist_f32_is_schedule_invariant() {
    pdist_case::<f32>(300, 130);
}

#[test]
fn pdist_f64_is_schedule_invariant() {
    pdist_case::<f64>(300, 130);
}

#[cfg(feature = "f16")]
mod half_floats {
    use super::*;

    /// Unit roundoff of f32, the precision the op computes in.
    const F32_UNIT: f64 = 1.0 / (1u64 << 24) as f64;

    /// Output format: unit roundoff and half the smallest subnormal step.
    struct Format {
        unit: f64,
        tiny: f64,
    }

    const F16: Format = Format {
        unit: 1.0 / (1u64 << 11) as f64,
        tiny: 1.0 / (1u64 << 25) as f64,
    };

    // bf16 shares f32's exponent range, so its subnormal step is below any
    // value these inputs produce.
    const BF16: Format = Format {
        unit: 1.0 / (1u64 << 8) as f64,
        tiny: 0.0,
    };

    /// f64 distance on inputs already rounded to the format, and the scale
    /// of the f32 rounding error the op's own computation can carry.
    fn reference(a: &[f64], b: &[f64], metric: DistanceMetric) -> (f64, f64) {
        let diffs = a.iter().zip(b).map(|(x, y)| x - y);
        match metric {
            DistanceMetric::SquaredEuclidean => {
                let s: f64 = diffs.map(|t| t * t).sum();
                (s, s)
            }
            DistanceMetric::Euclidean => {
                let s: f64 = diffs.map(|t| t * t).sum::<f64>().sqrt();
                (s, s)
            }
            DistanceMetric::Manhattan => {
                let s: f64 = diffs.map(f64::abs).sum();
                (s, s)
            }
            DistanceMetric::Chebyshev => (diffs.map(f64::abs).fold(0.0, f64::max), 0.0),
            DistanceMetric::Cosine => {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                // The f32 error lands on dot / (|a| |b|), whose size is at most 1.
                (1.0 - dot / (na * nb), 1.0)
            }
            other => panic!("no reference for {other:?}"),
        }
    }

    /// Bound: one rounding to the output format, plus the f32 error of a
    /// `d`-term reduction (`2 d` roundings: one per term, one per add).
    fn assert_close(got: f64, (want, scale): (f64, f64), d: usize, fmt: &Format, label: &str) {
        let f32_err = 2.0 * (d as f64 + 2.0) * F32_UNIT * scale;
        let tol = fmt.unit * (want.abs() + f32_err) + f32_err + fmt.tiny;
        assert!(
            (got - want).abs() <= tol,
            "{label}: got {got}, want {want}, tol {tol}"
        );
    }

    /// Input values as the format stores them, back in f64.
    fn rounded<T: Element>(data: &[f64]) -> Vec<f64> {
        data.iter().map(|&v| T::from_f64(v).to_f64()).collect()
    }

    fn check_cdist<T: Bits>(n: usize, m: usize, d: usize, fmt: &Format) {
        let xs = rounded::<T>(&values(n * d, 1));
        let ys = rounded::<T>(&values(m * d, 2));
        for (metric, out) in cdist_case::<T>(n, m, d) {
            for i in 0..n {
                for j in 0..m {
                    let a = &xs[i * d..(i + 1) * d];
                    let b = &ys[j * d..(j + 1) * d];
                    let label = format!("cdist {metric:?} {:?} ({i}, {j})", T::DTYPE);
                    assert_close(out[i * m + j], reference(a, b, metric), d, fmt, &label);
                }
            }
        }
    }

    fn check_pdist<T: Bits>(n: usize, d: usize, fmt: &Format) {
        let xs = rounded::<T>(&values(n * d, 3));
        for (metric, out) in pdist_case::<T>(n, d) {
            let mut k = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let a = &xs[i * d..(i + 1) * d];
                    let b = &xs[j * d..(j + 1) * d];
                    let label = format!("pdist {metric:?} {:?} ({i}, {j})", T::DTYPE);
                    assert_close(out[k], reference(a, b, metric), d, fmt, &label);
                    k += 1;
                }
            }
        }
    }

    #[test]
    fn cdist_f16_matches_reference_on_every_schedule() {
        check_cdist::<half::f16>(48, 130, 257, &F16);
        check_cdist::<half::f16>(1, 500, 257, &F16);
    }

    #[test]
    fn cdist_bf16_matches_reference_on_every_schedule() {
        check_cdist::<half::bf16>(48, 130, 257, &BF16);
        check_cdist::<half::bf16>(1, 500, 257, &BF16);
    }

    #[test]
    fn pdist_f16_matches_reference_on_every_schedule() {
        check_pdist::<half::f16>(300, 130, &F16);
    }

    #[test]
    fn pdist_bf16_matches_reference_on_every_schedule() {
        check_pdist::<half::bf16>(300, 130, &BF16);
    }
}
