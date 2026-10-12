//! Test data and an f64 host reference for the triangular-solve unit tests.
//!
//! The CPU, CUDA and WebGPU unit tests share these inputs so that every
//! backend runs the same cases.

/// Deterministic 64-bit LCG.
pub(crate) struct Lcg(pub u64);

impl Lcg {
    /// Uniform value in `[lo, hi)`.
    pub(crate) fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((self.0 >> 11) as f64) / ((1u64 << 53) as f64);
        lo + (hi - lo) * unit
    }
}

/// One shape case: batch dims, order `n`, and `Some(m)` columns or `None`
/// for a vector right-hand side.
pub(crate) struct Case {
    pub batch_dims: &'static [usize],
    pub n: usize,
    pub nrhs: Option<usize>,
}

impl Case {
    pub(crate) fn batch(&self) -> usize {
        self.batch_dims.iter().product()
    }

    pub(crate) fn m(&self) -> usize {
        self.nrhs.unwrap_or(1)
    }

    pub(crate) fn a_shape(&self) -> Vec<usize> {
        let mut s = self.batch_dims.to_vec();
        s.extend([self.n, self.n]);
        s
    }

    pub(crate) fn b_shape(&self) -> Vec<usize> {
        let mut s = self.batch_dims.to_vec();
        s.push(self.n);
        if let Some(m) = self.nrhs {
            s.push(m);
        }
        s
    }
}

/// Cases: no batch dims (vector and matrix RHS), one batch dim, two batch
/// dims, and an empty batch.
pub(crate) const CASES: &[Case] = &[
    Case {
        batch_dims: &[],
        n: 5,
        nrhs: None,
    },
    Case {
        batch_dims: &[],
        n: 5,
        nrhs: Some(3),
    },
    Case {
        batch_dims: &[4],
        n: 6,
        nrhs: Some(3),
    },
    Case {
        batch_dims: &[3],
        n: 5,
        nrhs: Some(1),
    },
    Case {
        batch_dims: &[2, 3],
        n: 4,
        nrhs: Some(2),
    },
    Case {
        batch_dims: &[0],
        n: 4,
        nrhs: Some(2),
    },
];

/// `batch` triangular `n x n` matrices, row-major and back to back.
///
/// The solved triangle holds off-diagonal values in `[-0.5, 0.5)` and a
/// diagonal in `[1, 2)`. The unread triangle holds 7.0. With `unit`, the
/// diagonal holds 5.0, which a solve must never read.
pub(crate) fn triangular_batch(
    seed: u64,
    batch: usize,
    n: usize,
    lower: bool,
    unit: bool,
) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(batch * n * n);
    for _ in 0..batch {
        for i in 0..n {
            for j in 0..n {
                let solved = if lower { j < i } else { j > i };
                let v = if i == j && unit {
                    5.0
                } else if i == j {
                    rng.uniform(1.0, 2.0)
                } else if solved {
                    rng.uniform(-0.5, 0.5)
                } else {
                    7.0
                };
                out.push(v);
            }
        }
    }
    out
}

/// `len` values in `[-1, 1)`.
pub(crate) fn values(seed: u64, len: usize) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..len).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// `I + A` for one `n x n` strictly lower `A` with entries in `[0.75, 0.85)`.
///
/// This is the shape of the gated delta rule's UT transform when keys
/// nearly coincide.
pub(crate) fn gdn_like_unit_lower(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
        for j in 0..i {
            out[i * n + j] = rng.uniform(0.75, 0.85);
        }
    }
    out
}

/// `n x n` identity, row-major.
pub(crate) fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

/// Host f64 substitution over a batch, as the reference result.
pub(crate) fn host_solve(
    a: &[f64],
    b: &[f64],
    batch: usize,
    n: usize,
    m: usize,
    lower: bool,
    unit: bool,
) -> Vec<f64> {
    let mut x = vec![0.0; batch * n * m];
    for bi in 0..batch {
        let a = &a[bi * n * n..(bi + 1) * n * n];
        let b = &b[bi * n * m..(bi + 1) * n * m];
        let x = &mut x[bi * n * m..(bi + 1) * n * m];
        for c in 0..m {
            for step in 0..n {
                let i = if lower { step } else { n - 1 - step };
                let range = if lower { 0..i } else { (i + 1)..n };
                let mut sum = b[i * m + c];
                for j in range {
                    sum -= a[i * n + j] * x[j * m + c];
                }
                x[i * m + c] = if unit { sum } else { sum / a[i * n + i] };
            }
        }
    }
    x
}

/// Largest absolute elementwise difference.
pub(crate) fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Narrow f64 values to f32.
pub(crate) fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Widen f32 values to f64.
pub(crate) fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}
