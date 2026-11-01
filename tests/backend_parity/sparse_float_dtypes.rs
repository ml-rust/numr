// CUDA sparse parity for the wider and narrower float dtypes: F64, and F16
// and BF16 under the `f16` feature. WebGPU holds 32-bit dtypes only, so
// `sparse_highlevel` covers its F64 refusal instead.
#![cfg(feature = "cuda")]

use crate::backend_parity::helpers::with_cuda_backend;
use crate::backend_parity::sparse_fixtures::*;
use crate::common::create_cpu_client;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::sparse::{CooData, CscData, CsrData, SparseOps, SparseTensor};
use numr::tensor::Tensor;

fn float_suite<R, C>(client: &C, device: &R::Device, dtype: DType) -> Snapshot
where
    R: Runtime<DType = DType>,
    C: SparseOps<R>,
{
    let t64 = |v: &[i64]| Tensor::<R>::from_slice(v, &[v.len()], device).unwrap();
    let tv = |v: &[f32]| float_tensor::<R>(v, dtype, device);
    let a_csr = SparseTensor::Csr(
        CsrData::new(t64(&A_ROW_PTRS), t64(&A_COLS), tv(&A_VALUES), SHAPE).unwrap(),
    );
    let b_csr = SparseTensor::Csr(
        CsrData::new(t64(&B_ROW_PTRS), t64(&B_COLS), tv(&B_VALUES), SHAPE).unwrap(),
    );
    let b_csc = SparseTensor::Csc(
        CscData::new(t64(&B_COL_PTRS), t64(&B_CSC_ROWS), tv(&B_CSC_VALUES), SHAPE).unwrap(),
    );
    let rows: Vec<i64> = A_SHUFFLED.iter().map(|e| e.0).collect();
    let cols: Vec<i64> = A_SHUFFLED.iter().map(|e| e.1).collect();
    let vals: Vec<f32> = A_SHUFFLED.iter().map(|e| e.2).collect();
    let a_coo = SparseTensor::Coo(CooData::new(t64(&rows), t64(&cols), tv(&vals), SHAPE).unwrap());
    let mut snap = Snapshot::new();

    // Low-level calls take the element type `T` of the values.
    macro_rules! typed {
        ($call:ident($($arg:expr),*)) => {
            (|| numr::dispatch_dtype!(dtype, T => { client.$call::<T>($($arg),*) }, "test"))()
                .unwrap()
        };
    }

    // Conversions.
    let (sr, sc, sv) = (t64(&rows), t64(&cols), tv(&vals));
    record_triple(
        &mut snap,
        "coo_to_csr",
        typed!(coo_to_csr(&sr, &sc, &sv, SHAPE)),
    );
    let (ap, ac, av) = (t64(&A_ROW_PTRS), t64(&A_COLS), tv(&A_VALUES));
    record_triple(
        &mut snap,
        "csr_to_csc",
        typed!(csr_to_csc(&ap, &ac, &av, SHAPE)),
    );
    let (cp, cr, cv) = (t64(&A_COL_PTRS), t64(&A_CSC_ROWS), tv(&A_CSC_VALUES));
    record_triple(
        &mut snap,
        "csc_to_coo",
        typed!(csc_to_coo(&cp, &cr, &cv, SHAPE)),
    );

    // Merges in every format.
    record_sparse(
        &mut snap,
        "add",
        &client.sparse_add(&a_csr, &b_csr).unwrap(),
    );
    record_sparse(
        &mut snap,
        "sub csc",
        &client.sparse_sub(&a_coo, &b_csc).unwrap(),
    );
    record_sparse(
        &mut snap,
        "mul",
        &client.sparse_mul(&a_csr, &b_csc).unwrap(),
    );
    let (bp, bc, bv) = (t64(&B_ROW_PTRS), t64(&B_COLS), tv(&B_VALUES));
    record_triple(
        &mut snap,
        "div_csr",
        typed!(div_csr(&ap, &ac, &av, &bp, &bc, &bv, SHAPE)),
    );
    let (dp, dr, dv) = (t64(&B_COL_PTRS), t64(&B_CSC_ROWS), tv(&B_CSC_VALUES));
    record_triple(
        &mut snap,
        "add_csc",
        typed!(add_csc(&cp, &cr, &cv, &dp, &dr, &dv, SHAPE)),
    );
    let (ar, br) = (t64(&A_ROWS), t64(&B_ROWS));
    record_triple(
        &mut snap,
        "sub_coo",
        typed!(sub_coo(&ar, &ac, &av, &br, &bc, &bv, SHAPE)),
    );

    // Products.
    let x = tv(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    snap.push((
        "spmv".into(),
        value_array(&client.spmv(&a_coo, &x).unwrap()),
    ));
    let dense_b = tv(&[1.0, 0.5, 2.0, -1.0, 0.0, 3.0, 1.5, 2.0, -2.0, 1.0])
        .reshape(&[5, 2])
        .unwrap();
    snap.push((
        "spmm".into(),
        value_array(&client.spmm(&a_csr, &dense_b).unwrap()),
    ));
    let dense_a = tv(&[1.0, 0.0, 2.0, -1.0, 0.5, 1.0, 0.0, 2.0])
        .reshape(&[2, 4])
        .unwrap();
    snap.push((
        "dsmm".into(),
        value_array(&client.dsmm(&dense_a, &a_csr).unwrap()),
    ));
    let b_t = client.sparse_transpose(&b_csr).unwrap();
    record_sparse(
        &mut snap,
        "matmul",
        &client.sparse_matmul(&a_csr, &b_t).unwrap(),
    );

    // Reductions, diagonal, scale and dense conversions.
    snap.push((
        "diag".into(),
        value_array(&client.sparse_extract_diagonal(&a_coo).unwrap()),
    ));
    snap.push((
        "sum".into(),
        value_array(&client.sparse_sum(&a_csr).unwrap()),
    ));
    snap.push((
        "sum_rows".into(),
        value_array(&client.sparse_sum_rows(&a_coo).unwrap()),
    ));
    snap.push((
        "sum_cols".into(),
        value_array(&client.sparse_sum_cols(&a_csr).unwrap()),
    ));
    record_sparse(
        &mut snap,
        "scale",
        &client.sparse_scale(&a_csr, 0.5).unwrap(),
    );
    let dense = client.sparse_to_dense(&a_coo).unwrap();
    snap.push(("to_dense".into(), value_array(&dense)));
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
    snap
}

fn check_dtype(dtype: DType, rtol: f64) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu = float_suite::<CpuRuntime, _>(&cpu_client, &cpu_device, dtype);
    with_cuda_backend(|client, device| {
        let gpu = float_suite::<CudaRuntime, _>(&client, &device, dtype);
        assert_same_snapshot(&cpu, &gpu, &format!("cuda {dtype:?}"), rtol);
    });
}

#[test]
fn cuda_f64_sparse_matches_the_cpu() {
    check_dtype(DType::F64, 1e-12);
}

/// F16 and BF16 round each result once. CUDA accumulates SpMV, SpMM and the
/// row and column sums in F32, so they can differ from the CPU by one ULP.
#[cfg(feature = "f16")]
#[test]
fn cuda_f16_sparse_matches_the_cpu() {
    check_dtype(DType::F16, 2e-3);
}

#[cfg(feature = "f16")]
#[test]
fn cuda_bf16_sparse_matches_the_cpu() {
    check_dtype(DType::BF16, 1.6e-2);
}

/// Output rows of 2000 entries over 2000 columns. Both exceed the fixed
/// shared-memory sizes an earlier SpGEMM kernel assumed.
#[test]
fn cuda_spgemm_matches_the_cpu_on_wide_rows() {
    const N: usize = 2000;
    fn operands<R: Runtime<DType = DType>>(
        device: &R::Device,
    ) -> (SparseTensor<R>, SparseTensor<R>) {
        // A is 2 x N with every entry stored: row r holds r + 1 + col / N.
        let a_ptrs: Vec<i64> = vec![0, N as i64, 2 * N as i64];
        let a_cols: Vec<i64> = (0..2).flat_map(|_| 0..N as i64).collect();
        let a_vals: Vec<f32> = (0..2 * N)
            .map(|i| (i / N + 1) as f32 + (i % N) as f32 / N as f32)
            .collect();
        // B is N x N with B[k, k] = 2 and B[k, (k + 7) % N] = -1.
        let b_ptrs: Vec<i64> = (0..=N as i64).map(|k| 2 * k).collect();
        let b_cols: Vec<i64> = (0..N as i64)
            .flat_map(|k| {
                let other = (k + 7) % N as i64;
                if other < k { [other, k] } else { [k, other] }
            })
            .collect();
        let b_vals: Vec<f32> = (0..N as i64)
            .flat_map(|k| {
                if (k + 7) % (N as i64) < k {
                    [-1.0, 2.0]
                } else {
                    [2.0, -1.0]
                }
            })
            .collect();
        let a = CsrData::<R>::from_slices(&a_ptrs, &a_cols, &a_vals, [2, N], device).unwrap();
        let b = CsrData::<R>::from_slices(&b_ptrs, &b_cols, &b_vals, [N, N], device).unwrap();
        (SparseTensor::Csr(a), SparseTensor::Csr(b))
    }

    let (cpu_client, cpu_device) = create_cpu_client();
    let (a, b) = operands::<CpuRuntime>(&cpu_device);
    let mut cpu = Snapshot::new();
    record_sparse(
        &mut cpu,
        "wide matmul",
        &cpu_client.sparse_matmul(&a, &b).unwrap(),
    );
    with_cuda_backend(|client, device| {
        let (a, b) = operands::<CudaRuntime>(&device);
        let mut gpu = Snapshot::new();
        record_sparse(
            &mut gpu,
            "wide matmul",
            &client.sparse_matmul(&a, &b).unwrap(),
        );
        assert_same_snapshot(&cpu, &gpu, "cuda", 1e-6);
    });
}

/// A dense operand of 1100 columns, more than one CUDA block holds threads.
#[test]
fn cuda_spmm_matches_the_cpu_past_1024_columns() {
    const COLS: usize = 1100;
    let b: Vec<f32> = (0..5 * COLS).map(|i| (i % 13) as f32 - 6.0).collect();
    let (cpu_client, cpu_device) = create_cpu_client();
    let csr = |device| {
        let ptrs = Tensor::<CpuRuntime>::from_slice(&A_ROW_PTRS, &[5], device).unwrap();
        let cols = Tensor::<CpuRuntime>::from_slice(&A_COLS, &[9], device).unwrap();
        let vals = Tensor::<CpuRuntime>::from_slice(&A_VALUES, &[9], device).unwrap();
        SparseTensor::Csr(CsrData::new(ptrs, cols, vals, SHAPE).unwrap())
    };
    let b_cpu = Tensor::<CpuRuntime>::from_slice(&b, &[5, COLS], &cpu_device).unwrap();
    let want: Vec<f32> = cpu_client.spmm(&csr(&cpu_device), &b_cpu).unwrap().to_vec();
    with_cuda_backend(|client, device| {
        let a =
            CsrData::<CudaRuntime>::from_slices(&A_ROW_PTRS, &A_COLS, &A_VALUES, SHAPE, &device)
                .unwrap();
        let b_gpu = Tensor::<CudaRuntime>::from_slice(&b, &[5, COLS], &device).unwrap();
        let got: Vec<f32> = client.spmm(&SparseTensor::Csr(a), &b_gpu).unwrap().to_vec();
        assert_eq!(got, want, "cuda spmm past 1024 columns");
    });
}
