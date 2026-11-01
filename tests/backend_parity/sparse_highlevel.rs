// Backend parity for the high-level sparse entry points that take a
// `SparseTensor` and run the backend's low-level kernel. CSR indices are I64
// on every runtime, so each backend must hand its kernel indices it can read.

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::backend_parity::sparse_fixtures::*;
use crate::common::create_cpu_client;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::CudaRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
use numr::sparse::{CooData, CscData, CsrData, SparseOps, SparseTensor};
use numr::tensor::Tensor;

/// The matrix
///
/// ```text
/// [1 0 2]
/// [0 3 0]
/// [4 0 5]
/// ```
fn csr_3x3<R: Runtime<DType = DType>>(device: &R::Device) -> CsrData<R> {
    let row_ptrs = Tensor::<R>::from_slice(&[0i64, 2, 3, 5], &[4], device).unwrap();
    let col_indices = Tensor::<R>::from_slice(&[0i64, 2, 1, 0, 2], &[5], device).unwrap();
    let values = Tensor::<R>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], device).unwrap();
    CsrData::new(row_ptrs, col_indices, values, [3, 3]).unwrap()
}

/// The same matrix in COO form.
fn coo_3x3<R: Runtime<DType = DType>>(device: &R::Device) -> CooData<R> {
    let rows = Tensor::<R>::from_slice(&[0i64, 0, 1, 2, 2], &[5], device).unwrap();
    let cols = Tensor::<R>::from_slice(&[0i64, 2, 1, 0, 2], &[5], device).unwrap();
    let values = Tensor::<R>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], device).unwrap();
    CooData::new(rows, cols, values, [3, 3]).unwrap()
}

const X: [f32; 3] = [1.0, 2.0, 3.0];
const SPMV: [f32; 3] = [7.0, 6.0, 19.0];

/// `B` is 3x2 row-major: `[[1, 2], [3, 4], [5, 6]]`.
const B: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const SPMM: [f32; 6] = [11.0, 14.0, 9.0, 12.0, 29.0, 38.0];

#[test]
fn spmv_on_a_csr_matrix_matches_the_reference() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let x = Tensor::<CpuRuntime>::from_slice(&X, &[3], &cpu_device).unwrap();
    let a = SparseTensor::Csr(csr_3x3::<CpuRuntime>(&cpu_device));
    let y: Vec<f32> = cpu_client.spmv(&a, &x).unwrap().to_vec();
    assert_eq!(y, SPMV, "cpu");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let x = Tensor::<CudaRuntime>::from_slice(&X, &[3], &device).unwrap();
        let a = SparseTensor::Csr(csr_3x3::<CudaRuntime>(&device));
        let y: Vec<f32> = client.spmv(&a, &x).unwrap().to_vec();
        assert_eq!(y, SPMV, "cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let x = Tensor::<WgpuRuntime>::from_slice(&X, &[3], &device).unwrap();
        let a = SparseTensor::Csr(csr_3x3::<WgpuRuntime>(&device));
        let y: Vec<f32> = client.spmv(&a, &x).unwrap().to_vec();
        assert_eq!(y, SPMV, "wgpu");
    });
}

#[test]
fn spmv_on_a_coo_matrix_matches_the_reference() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let x = Tensor::<CpuRuntime>::from_slice(&X, &[3], &cpu_device).unwrap();
    let a = SparseTensor::Coo(coo_3x3::<CpuRuntime>(&cpu_device));
    let y: Vec<f32> = cpu_client.spmv(&a, &x).unwrap().to_vec();
    assert_eq!(y, SPMV, "cpu");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let x = Tensor::<CudaRuntime>::from_slice(&X, &[3], &device).unwrap();
        let a = SparseTensor::Coo(coo_3x3::<CudaRuntime>(&device));
        let y: Vec<f32> = client.spmv(&a, &x).unwrap().to_vec();
        assert_eq!(y, SPMV, "cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let x = Tensor::<WgpuRuntime>::from_slice(&X, &[3], &device).unwrap();
        let a = SparseTensor::Coo(coo_3x3::<WgpuRuntime>(&device));
        let y: Vec<f32> = client.spmv(&a, &x).unwrap().to_vec();
        assert_eq!(y, SPMV, "wgpu");
    });
}

#[test]
fn spmm_on_a_csr_matrix_matches_the_reference() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let b = Tensor::<CpuRuntime>::from_slice(&B, &[3, 2], &cpu_device).unwrap();
    let a = SparseTensor::Csr(csr_3x3::<CpuRuntime>(&cpu_device));
    let c: Vec<f32> = cpu_client.spmm(&a, &b).unwrap().to_vec();
    assert_eq!(c, SPMM, "cpu");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let b = Tensor::<CudaRuntime>::from_slice(&B, &[3, 2], &device).unwrap();
        let a = SparseTensor::Csr(csr_3x3::<CudaRuntime>(&device));
        let c: Vec<f32> = client.spmm(&a, &b).unwrap().to_vec();
        assert_eq!(c, SPMM, "cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let b = Tensor::<WgpuRuntime>::from_slice(&B, &[3, 2], &device).unwrap();
        let a = SparseTensor::Csr(csr_3x3::<WgpuRuntime>(&device));
        let c: Vec<f32> = client.spmm(&a, &b).unwrap().to_vec();
        assert_eq!(c, SPMM, "wgpu");
    });
}

/// `A + A` doubles every stored value and keeps the pattern.
#[test]
fn adding_a_matrix_to_itself_doubles_its_values() {
    let doubled: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0, 5.0]
        .iter()
        .map(|v| v * 2.0)
        .collect();

    let (cpu_client, cpu_device) = create_cpu_client();
    let a = SparseTensor::Csr(csr_3x3::<CpuRuntime>(&cpu_device));
    let sum = cpu_client.sparse_add(&a, &a).unwrap();
    let SparseTensor::Csr(sum) = sum else {
        panic!("a CSR sum is CSR");
    };
    let values: Vec<f32> = sum.values().to_vec();
    assert_eq!(values, doubled, "cpu");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let a = SparseTensor::Csr(csr_3x3::<CudaRuntime>(&device));
        let SparseTensor::Csr(sum) = client.sparse_add(&a, &a).unwrap() else {
            panic!("a CSR sum is CSR");
        };
        let values: Vec<f32> = sum.values().to_vec();
        assert_eq!(values, doubled, "cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let a = SparseTensor::Csr(csr_3x3::<WgpuRuntime>(&device));
        let SparseTensor::Csr(sum) = client.sparse_add(&a, &a).unwrap() else {
            panic!("a CSR sum is CSR");
        };
        let values: Vec<f32> = sum.values().to_vec();
        assert_eq!(values, doubled, "wgpu");
    });
}

// ---------------------------------------------------------------------------
// Every `SparseOps` entry point, CPU against each GPU backend.
//
// `sparse_suite` runs the same calls on any runtime and copies every output
// array to the host. Index arrays must come back I64 and match exactly. Values
// match within a few ULPs, because WGSL division is not correctly rounded.
// ---------------------------------------------------------------------------

fn sparse_suite<R, C>(client: &C, device: &R::Device) -> Snapshot
where
    R: Runtime<DType = DType>,
    C: SparseOps<R>,
{
    let t64 = |v: &[i64]| Tensor::<R>::from_slice(v, &[v.len()], device).unwrap();
    let t32 = |v: &[f32]| Tensor::<R>::from_slice(v, &[v.len()], device).unwrap();
    let csr = |p: &[i64], i: &[i64], v: &[f32]| {
        SparseTensor::Csr(CsrData::new(t64(p), t64(i), t32(v), SHAPE).unwrap())
    };
    let a_csr = csr(&A_ROW_PTRS, &A_COLS, &A_VALUES);
    let b_csr = csr(&B_ROW_PTRS, &B_COLS, &B_VALUES);
    let a_csc = SparseTensor::Csc(
        CscData::new(
            t64(&A_COL_PTRS),
            t64(&A_CSC_ROWS),
            t32(&A_CSC_VALUES),
            SHAPE,
        )
        .unwrap(),
    );
    let shuffled_rows: Vec<i64> = A_SHUFFLED.iter().map(|e| e.0).collect();
    let shuffled_cols: Vec<i64> = A_SHUFFLED.iter().map(|e| e.1).collect();
    let shuffled_values: Vec<f32> = A_SHUFFLED.iter().map(|e| e.2).collect();
    let a_coo = SparseTensor::Coo(
        CooData::new(
            t64(&shuffled_rows),
            t64(&shuffled_cols),
            t32(&shuffled_values),
            SHAPE,
        )
        .unwrap(),
    );
    let mut snap = Snapshot::new();

    // Format conversions, low level. The COO input is shuffled.
    let (sr, sc, sv) = (
        t64(&shuffled_rows),
        t64(&shuffled_cols),
        t32(&shuffled_values),
    );
    record_triple(
        &mut snap,
        "coo_to_csr",
        client.coo_to_csr::<f32>(&sr, &sc, &sv, SHAPE).unwrap(),
    );
    record_triple(
        &mut snap,
        "coo_to_csc",
        client.coo_to_csc::<f32>(&sr, &sc, &sv, SHAPE).unwrap(),
    );
    let (ap, ac, av) = (t64(&A_ROW_PTRS), t64(&A_COLS), t32(&A_VALUES));
    record_triple(
        &mut snap,
        "csr_to_csc",
        client.csr_to_csc::<f32>(&ap, &ac, &av, SHAPE).unwrap(),
    );
    record_triple(
        &mut snap,
        "csr_to_coo",
        client.csr_to_coo::<f32>(&ap, &ac, &av, SHAPE).unwrap(),
    );
    let (cp, cr, cv) = (t64(&A_COL_PTRS), t64(&A_CSC_ROWS), t32(&A_CSC_VALUES));
    record_triple(
        &mut snap,
        "csc_to_csr",
        client.csc_to_csr::<f32>(&cp, &cr, &cv, SHAPE).unwrap(),
    );
    record_triple(
        &mut snap,
        "csc_to_coo",
        client.csc_to_coo::<f32>(&cp, &cr, &cv, SHAPE).unwrap(),
    );

    // Duplicates keep their input order inside a row.
    let dup = client
        .coo_to_csr::<f32>(
            &t64(&[1, 0, 1, 0]),
            &t64(&[2, 1, 2, 1]),
            &t32(&[1.0, 2.0, 3.0, 4.0]),
            [2, 3],
        )
        .unwrap();
    record_triple(&mut snap, "coo_to_csr duplicates", dup);

    // Round trip CSR -> CSC -> CSR through the high-level API.
    let round_trip = client
        .dense_to_csr(&client.sparse_to_dense(&a_csc).unwrap(), 0.5)
        .unwrap();
    record_sparse(
        &mut snap,
        "dense_to_csr(csc A)",
        &SparseTensor::Csr(round_trip),
    );

    // Merges, high level (CSR result) and low level for each format.
    record_sparse(
        &mut snap,
        "sparse_add",
        &client.sparse_add(&a_csr, &b_csr).unwrap(),
    );
    record_sparse(
        &mut snap,
        "sparse_sub",
        &client.sparse_sub(&a_csr, &b_csr).unwrap(),
    );
    record_sparse(
        &mut snap,
        "sparse_mul",
        &client.sparse_mul(&a_coo, &b_csr).unwrap(),
    );
    record_sparse(
        &mut snap,
        "sparse_add coo+csc",
        &client.sparse_add(&a_coo, &a_csc).unwrap(),
    );
    let (bp, bc, bv) = (t64(&B_ROW_PTRS), t64(&B_COLS), t32(&B_VALUES));
    record_triple(
        &mut snap,
        "div_csr",
        client
            .div_csr::<f32>(&ap, &ac, &av, &bp, &bc, &bv, SHAPE)
            .unwrap(),
    );
    let (dp, dr, dv) = (t64(&B_COL_PTRS), t64(&B_CSC_ROWS), t32(&B_CSC_VALUES));
    record_triple(
        &mut snap,
        "add_csc",
        client
            .add_csc::<f32>(&cp, &cr, &cv, &dp, &dr, &dv, SHAPE)
            .unwrap(),
    );
    record_triple(
        &mut snap,
        "sub_csc",
        client
            .sub_csc::<f32>(&cp, &cr, &cv, &dp, &dr, &dv, SHAPE)
            .unwrap(),
    );
    record_triple(
        &mut snap,
        "mul_csc",
        client
            .mul_csc::<f32>(&cp, &cr, &cv, &dp, &dr, &dv, SHAPE)
            .unwrap(),
    );
    record_triple(
        &mut snap,
        "div_csc",
        client
            .div_csc::<f32>(&cp, &cr, &cv, &dp, &dr, &dv, SHAPE)
            .unwrap(),
    );
    let (ar, br) = (t64(&A_ROWS), t64(&B_ROWS));
    record_triple(
        &mut snap,
        "add_coo",
        client
            .add_coo::<f32>(&ar, &ac, &av, &br, &bc, &bv, SHAPE)
            .unwrap(),
    );
    record_triple(
        &mut snap,
        "sub_coo",
        client
            .sub_coo::<f32>(&ar, &ac, &av, &br, &bc, &bv, SHAPE)
            .unwrap(),
    );
    record_triple(
        &mut snap,
        "mul_coo",
        client
            .mul_coo::<f32>(&ar, &ac, &av, &br, &bc, &bv, SHAPE)
            .unwrap(),
    );
    record_triple(
        &mut snap,
        "div_coo",
        client
            .div_coo::<f32>(&ar, &ac, &av, &br, &bc, &bv, SHAPE)
            .unwrap(),
    );

    // A - A cancels every entry. Explicit zeros can stay stored, so compare dense.
    let cancelled = client.sparse_sub(&a_csr, &a_csr).unwrap();
    snap.push((
        "A - A dense".into(),
        value_array(&client.sparse_to_dense(&cancelled).unwrap()),
    ));

    // An empty operand.
    let empty = SparseTensor::Csr(CsrData::<R>::empty(SHAPE, DType::F32, device).unwrap());
    record_sparse(
        &mut snap,
        "empty + B",
        &client.sparse_add(&empty, &b_csr).unwrap(),
    );
    record_sparse(
        &mut snap,
        "A * empty",
        &client.sparse_mul(&a_csr, &empty).unwrap(),
    );
    let x = t32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    snap.push((
        "spmv empty".into(),
        value_array(&client.spmv(&empty, &x).unwrap()),
    ));

    // Diagonal: (1,1) and (2,2) are not stored.
    snap.push((
        "diag csr".into(),
        value_array(&client.sparse_extract_diagonal(&a_csr).unwrap()),
    ));
    snap.push((
        "diag coo".into(),
        value_array(&client.sparse_extract_diagonal(&a_coo).unwrap()),
    ));

    // Scalar ops, reductions, dense conversions, transpose.
    record_sparse(
        &mut snap,
        "scale",
        &client.sparse_scale(&a_coo, 2.5).unwrap(),
    );
    snap.push((
        "sum".into(),
        value_array(&client.sparse_sum(&a_csc).unwrap()),
    ));
    snap.push((
        "sum_rows".into(),
        value_array(&client.sparse_sum_rows(&a_coo).unwrap()),
    ));
    snap.push((
        "sum_cols".into(),
        value_array(&client.sparse_sum_cols(&a_csr).unwrap()),
    ));
    snap.push((
        "nnz_per_row".into(),
        index_array(&client.sparse_nnz_per_row(&a_csc).unwrap(), "nnz_per_row"),
    ));
    snap.push((
        "nnz_per_col".into(),
        index_array(&client.sparse_nnz_per_col(&a_coo).unwrap(), "nnz_per_col"),
    ));
    snap.push((
        "spmv coo".into(),
        value_array(&client.spmv(&a_coo, &x).unwrap()),
    ));
    snap.push((
        "spmv csc".into(),
        value_array(&client.spmv(&a_csc, &x).unwrap()),
    ));
    let dense = client.sparse_to_dense(&a_coo).unwrap();
    snap.push(("to_dense coo".into(), value_array(&dense)));
    record_sparse(
        &mut snap,
        "dense_to_sparse 4.5",
        &client.dense_to_sparse(&dense, 4.5).unwrap(),
    );
    record_sparse(
        &mut snap,
        "dense_to_csc 0.5",
        &SparseTensor::Csc(client.dense_to_csc(&dense, 0.5).unwrap()),
    );
    let transposed = client.sparse_transpose(&a_csr).unwrap();
    record_sparse(&mut snap, "transpose csr", &transposed);
    snap.push((
        "transpose dense".into(),
        value_array(&client.sparse_to_dense(&transposed).unwrap()),
    ));

    // I32 values merge through the typed kernels.
    let ti = |v: &[i32]| Tensor::<R>::from_slice(v, &[v.len()], device).unwrap();
    let ai = SparseTensor::Csr(
        CsrData::new(
            t64(&A_ROW_PTRS),
            t64(&A_COLS),
            ti(&[1, 2, 3, 5, 4, 6, 9, 7, 8]),
            SHAPE,
        )
        .unwrap(),
    );
    let bi = SparseTensor::Csr(
        CsrData::new(
            t64(&B_ROW_PTRS),
            t64(&B_COLS),
            ti(&[10, 20, 30, 40, 50, 60]),
            SHAPE,
        )
        .unwrap(),
    );
    record_sparse(
        &mut snap,
        "i32 sparse_sub",
        &client.sparse_sub(&ai, &bi).unwrap(),
    );
    snap
}

#[test]
fn every_sparse_entry_point_matches_the_cpu() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu = sparse_suite::<CpuRuntime, _>(&cpu_client, &cpu_device);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let gpu = sparse_suite::<CudaRuntime, _>(&client, &device);
        assert_same_snapshot(&cpu, &gpu, "cuda", 1e-6);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let gpu = sparse_suite::<WgpuRuntime, _>(&client, &device);
        assert_same_snapshot(&cpu, &gpu, "wgpu", 1e-6);
    });
}

/// WebGPU holds 32-bit dtypes only. A wider value dtype returns
/// `UnsupportedDType`, never a panic inside wgpu.
#[cfg(feature = "wgpu")]
#[test]
fn webgpu_refuses_f64_sparse_values() {
    use numr::error::Error;

    with_wgpu_backend(|client, device| {
        let csr = || {
            let p = Tensor::<WgpuRuntime>::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
            let i = Tensor::<WgpuRuntime>::from_slice(&[0i64, 1], &[2], &device).unwrap();
            let v = Tensor::<WgpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device).unwrap();
            SparseTensor::Csr(CsrData::new(p, i, v, [2, 2]).unwrap())
        };
        let x = Tensor::<WgpuRuntime>::from_slice(&[1.0f64, 1.0], &[2], &device).unwrap();
        let checks = [
            ("spmv", client.spmv(&csr(), &x).map(|_| ())),
            ("sparse_add", client.sparse_add(&csr(), &csr()).map(|_| ())),
            (
                "sparse_to_dense",
                client.sparse_to_dense(&csr()).map(|_| ()),
            ),
            (
                "sparse_transpose then to_csr",
                client
                    .sparse_add(&client.sparse_transpose(&csr()).unwrap(), &csr())
                    .map(|_| ()),
            ),
            (
                "sparse_extract_diagonal",
                client.sparse_extract_diagonal(&csr()).map(|_| ()),
            ),
        ];
        for (name, result) in checks {
            match result {
                Err(Error::UnsupportedDType { dtype, .. }) => {
                    assert_eq!(dtype, DType::F64, "{name}")
                }
                other => panic!("{name}: want UnsupportedDType, got {other:?}"),
            }
        }
    });
}
