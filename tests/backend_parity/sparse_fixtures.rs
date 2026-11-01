// Shared fixtures and host snapshots for the sparse parity suites.
//
// A suite runs the same calls on any runtime and copies every output array to
// the host. Index arrays must come back I64 and match exactly. Values match
// within a relative tolerance chosen per dtype.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::sparse::SparseTensor;
use numr::tensor::Tensor;

/// One output array copied to the host.
#[derive(Debug)]
pub enum Array {
    Index(Vec<i64>),
    Value(Vec<f64>),
}

pub type Snapshot = Vec<(String, Array)>;

pub fn index_array<R: Runtime<DType = DType>>(t: &Tensor<R>, what: &str) -> Array {
    assert_eq!(t.dtype(), DType::I64, "{what}: index dtype");
    Array::Index(t.to_vec::<i64>())
}

pub fn value_array<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Array {
    let values: Vec<f64> = match t.dtype() {
        DType::F64 => t.to_vec::<f64>(),
        DType::F32 => t.to_vec::<f32>().iter().map(|&v| f64::from(v)).collect(),
        #[cfg(feature = "f16")]
        DType::F16 => t.to_vec::<half::f16>().iter().map(|v| v.to_f64()).collect(),
        #[cfg(feature = "f16")]
        DType::BF16 => t
            .to_vec::<half::bf16>()
            .iter()
            .map(|v| v.to_f64())
            .collect(),
        DType::I32 => t.to_vec::<i32>().iter().map(|&v| f64::from(v)).collect(),
        DType::I64 => t.to_vec::<i64>().iter().map(|&v| v as f64).collect(),
        dtype => panic!("no host copy for {dtype:?}"),
    };
    Array::Value(values)
}

/// A device tensor of `values` converted on the host to `dtype`.
pub fn float_tensor<R: Runtime<DType = DType>>(
    values: &[f32],
    dtype: DType,
    device: &R::Device,
) -> Tensor<R> {
    let shape = [values.len()];
    match dtype {
        DType::F32 => Tensor::<R>::from_slice(values, &shape, device),
        DType::F64 => {
            let v: Vec<f64> = values.iter().map(|&x| f64::from(x)).collect();
            Tensor::<R>::from_slice(&v, &shape, device)
        }
        #[cfg(feature = "f16")]
        DType::F16 => {
            let v: Vec<half::f16> = values.iter().map(|&x| half::f16::from_f32(x)).collect();
            Tensor::<R>::from_slice(&v, &shape, device)
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            let v: Vec<half::bf16> = values.iter().map(|&x| half::bf16::from_f32(x)).collect();
            Tensor::<R>::from_slice(&v, &shape, device)
        }
        _ => panic!("no float fixture for {dtype:?}"),
    }
    .unwrap()
}

/// Record the three arrays of a sparse result, and its format and shape.
pub fn record_sparse<R: Runtime<DType = DType>>(
    snap: &mut Snapshot,
    name: &str,
    t: &SparseTensor<R>,
) {
    let (format, i0, i1, values) = match t {
        SparseTensor::Csr(d) => ("csr", d.row_ptrs(), d.col_indices(), d.values()),
        SparseTensor::Csc(d) => ("csc", d.col_ptrs(), d.row_indices(), d.values()),
        SparseTensor::Coo(d) => ("coo", d.row_indices(), d.col_indices(), d.values()),
    };
    let [nrows, ncols] = t.shape();
    snap.push((
        format!("{name} {format} {nrows}x{ncols}"),
        Array::Index(vec![nrows as i64, ncols as i64]),
    ));
    snap.push((format!("{name} i0"), index_array(i0, name)));
    snap.push((format!("{name} i1"), index_array(i1, name)));
    snap.push((format!("{name} values"), value_array(values)));
}

/// Record a `(indices, indices, values)` triple from a low-level method.
pub fn record_triple<R: Runtime<DType = DType>>(
    snap: &mut Snapshot,
    name: &str,
    (i0, i1, values): (Tensor<R>, Tensor<R>, Tensor<R>),
) {
    snap.push((format!("{name} i0"), index_array(&i0, name)));
    snap.push((format!("{name} i1"), index_array(&i1, name)));
    snap.push((format!("{name} values"), value_array(&values)));
}

/// Check that `gpu` equals `cpu`: indices exactly, values within `rtol`
/// relative to `max(|want|, 1)`.
pub fn assert_same_snapshot(cpu: &Snapshot, gpu: &Snapshot, backend: &str, rtol: f64) {
    assert_eq!(cpu.len(), gpu.len(), "{backend}: snapshot length");
    for ((name, want), (gpu_name, got)) in cpu.iter().zip(gpu) {
        assert_eq!(name, gpu_name, "{backend}: snapshot order");
        match (want, got) {
            (Array::Index(w), Array::Index(g)) => assert_eq!(w, g, "{backend}: {name}"),
            (Array::Value(w), Array::Value(g)) => {
                assert_eq!(w.len(), g.len(), "{backend}: {name} length");
                for (i, (w, g)) in w.iter().zip(g).enumerate() {
                    let tol = rtol * w.abs().max(1.0);
                    assert!(
                        (w - g).abs() <= tol,
                        "{backend}: {name}[{i}] = {g}, want {w}"
                    );
                }
            }
            _ => panic!("{backend}: {name}: array kinds differ ({want:?} vs {got:?})"),
        }
    }
}

/// `A` is 4x5:
///
/// ```text
/// [1 2 0 3 0]
/// [0 0 5 0 4]
/// [6 0 0 0 9]
/// [7 0 0 8 0]
/// ```
pub const A_ROW_PTRS: [i64; 5] = [0, 3, 5, 7, 9];
pub const A_COLS: [i64; 9] = [0, 1, 3, 2, 4, 0, 4, 0, 3];
pub const A_ROWS: [i64; 9] = [0, 0, 0, 1, 1, 2, 2, 3, 3];
pub const A_VALUES: [f32; 9] = [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 9.0, 7.0, 8.0];
pub const A_COL_PTRS: [i64; 6] = [0, 3, 4, 5, 7, 9];
pub const A_CSC_ROWS: [i64; 9] = [0, 2, 3, 0, 1, 0, 3, 1, 2];
pub const A_CSC_VALUES: [f32; 9] = [1.0, 6.0, 7.0, 2.0, 5.0, 3.0, 8.0, 4.0, 9.0];
/// `A` as COO in shuffled order.
pub const A_SHUFFLED: [(i64, i64, f32); 9] = [
    (2, 4, 9.0),
    (0, 1, 2.0),
    (3, 0, 7.0),
    (0, 3, 3.0),
    (1, 2, 5.0),
    (2, 0, 6.0),
    (0, 0, 1.0),
    (3, 3, 8.0),
    (1, 4, 4.0),
];

/// `B` is 4x5 and overlaps `A` at (0,1), (1,2), (2,4) and (3,3):
///
/// ```text
/// [0 10 20  0  0]
/// [0  0 30  0  0]
/// [0  0  0  0 40]
/// [0 50  0 60  0]
/// ```
pub const B_ROW_PTRS: [i64; 5] = [0, 2, 3, 4, 6];
pub const B_COLS: [i64; 6] = [1, 2, 2, 4, 1, 3];
pub const B_ROWS: [i64; 6] = [0, 0, 1, 2, 3, 3];
pub const B_VALUES: [f32; 6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
pub const B_COL_PTRS: [i64; 6] = [0, 0, 2, 4, 5, 6];
pub const B_CSC_ROWS: [i64; 6] = [0, 3, 0, 1, 3, 2];
pub const B_CSC_VALUES: [f32; 6] = [10.0, 50.0, 20.0, 30.0, 60.0, 40.0];

pub const SHAPE: [usize; 2] = [4, 5];
