// Backend parity tests for BinaryOps trait
//
// Canonical pattern:
// - BinaryOp enum
// - apply_binary_op dispatcher
// - shared test_binary_parity runner
// - tiny per-op tests via macro

use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::backend_parity::helpers::assert_case_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
    Atan2,
}

#[derive(Clone)]
struct TestCase {
    a: Vec<f32>,
    a_shape: Vec<usize>,
    b: Vec<f32>,
    b_shape: Vec<usize>,
}

impl TestCase {
    fn new(a: Vec<f32>, a_shape: Vec<usize>, b: Vec<f32>, b_shape: Vec<usize>) -> Self {
        Self {
            a,
            a_shape,
            b,
            b_shape,
        }
    }
}

fn apply_binary_op<R: Runtime>(
    client: &impl BinaryOps<R>,
    op: BinaryOp,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        BinaryOp::Add => client.add(a, b),
        BinaryOp::Sub => client.sub(a, b),
        BinaryOp::Mul => client.mul(a, b),
        BinaryOp::Div => client.div(a, b),
        BinaryOp::Pow => client.pow(a, b),
        BinaryOp::Maximum => client.maximum(a, b),
        BinaryOp::Minimum => client.minimum(a, b),
        BinaryOp::Atan2 => client.atan2(a, b),
    }
}

fn test_binary_parity(op: BinaryOp, test_cases: &[TestCase]) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_results: Vec<Vec<f32>> = test_cases
        .iter()
        .map(|tc| {
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &cpu_device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &cpu_device);
            apply_binary_op(&cpu_client, op, &a, &b)
                .expect("CPU operation failed")
                .to_vec::<f32>()
        })
        .collect();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &cuda_device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &cuda_device);
            let cuda_result = apply_binary_op(&cuda_client, op, &a, &b)
                .expect("CUDA operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &cuda_result, &format!("{op:?}"), "cuda");
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &wgpu_device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &wgpu_device);
            let wgpu_result = apply_binary_op(&wgpu_client, op, &a, &b)
                .expect("WebGPU operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &wgpu_result, &format!("{op:?}"), "wgpu");
        }
    });
}

macro_rules! binary_case {
    ($name:ident, $op:expr, $cases:expr) => {
        #[test]
        fn $name() {
            test_binary_parity($op, $cases);
        }
    };
}

binary_case!(
    test_add_parity,
    BinaryOp::Add,
    &[
        TestCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![4]
        ),
        TestCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![2, 2]
        ),
        TestCase::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![10.0], vec![1]),
        TestCase::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![5.0], vec![]),
    ]
);

binary_case!(
    test_sub_parity,
    BinaryOp::Sub,
    &[
        TestCase::new(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![4],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4]
        ),
        TestCase::new(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![2, 2],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2, 2]
        ),
    ]
);

binary_case!(
    test_mul_parity,
    BinaryOp::Mul,
    &[
        TestCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![4]
        ),
        TestCase::new(
            vec![0.5, 1.5, 2.5, 3.5],
            vec![2, 2],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![2, 2]
        ),
        TestCase::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![2.0], vec![]),
    ]
);

binary_case!(
    test_div_parity,
    BinaryOp::Div,
    &[
        TestCase::new(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![4],
            vec![2.0, 4.0, 5.0, 8.0],
            vec![4]
        ),
        TestCase::new(
            vec![100.0, 200.0, 300.0, 400.0],
            vec![2, 2],
            vec![2.0, 4.0, 5.0, 8.0],
            vec![2, 2],
        ),
    ]
);

binary_case!(
    test_pow_parity,
    BinaryOp::Pow,
    &[
        TestCase::new(
            vec![2.0, 3.0, 4.0, 5.0],
            vec![4],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![4]
        ),
        TestCase::new(
            vec![2.0, 3.0, 4.0, 5.0],
            vec![2, 2],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![2, 2]
        ),
    ]
);

binary_case!(
    test_maximum_parity,
    BinaryOp::Maximum,
    &[
        TestCase::new(
            vec![1.0, 5.0, 3.0, 2.0],
            vec![4],
            vec![3.0, 2.0, 5.0, 1.0],
            vec![4]
        ),
        TestCase::new(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![2, 2],
            vec![15.0, 15.0, 15.0, 15.0],
            vec![2, 2],
        ),
    ]
);

binary_case!(
    test_minimum_parity,
    BinaryOp::Minimum,
    &[
        TestCase::new(
            vec![1.0, 5.0, 3.0, 2.0],
            vec![4],
            vec![3.0, 2.0, 5.0, 1.0],
            vec![4]
        ),
        TestCase::new(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![2, 2],
            vec![15.0, 15.0, 15.0, 15.0],
            vec![2, 2],
        ),
    ]
);

binary_case!(
    test_atan2_parity,
    BinaryOp::Atan2,
    &[
        TestCase::new(
            vec![0.0, 1.0, 1.0, 0.0],
            vec![4],
            vec![1.0, 0.0, 1.0, 1.0],
            vec![4]
        ),
        TestCase::new(
            vec![1.0, -1.0, -1.0, 1.0],
            vec![2, 2],
            vec![1.0, 1.0, -1.0, -1.0],
            vec![2, 2]
        ),
    ]
);
