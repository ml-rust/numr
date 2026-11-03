// CUDA and WebGPU parity for `conv2d_fused`, against the CPU result.

use numr::ops::ConvLayout;

use super::{ACTS, CASES, fused, fused_scaled, operands};
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose_tol, create_cpu_client, gemm_long_k_tolerance,
    is_dtype_supported, parity_dtypes,
};

#[test]
fn test_conv2d_fused_backend_parity() {
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        for case in CASES {
            let cpu_ops = operands(&cpu_client, &cpu_device, dtype, case);
            let (rtol, atol) = gemm_long_k_tolerance(dtype, case.contraction(), 1.0);
            for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
                for act in ACTS {
                    let cpu = fused(&cpu_client, &cpu_ops, case, layout, act, true, true);
                    let label = format!("{} {layout:?} {act:?} [{dtype:?}]", case.name);

                    #[cfg(feature = "cuda")]
                    if is_dtype_supported("cuda", dtype) {
                        with_cuda_backend(|client, device| {
                            let ops = operands(&client, &device, dtype, case);
                            let got = fused(&client, &ops, case, layout, act, true, true);
                            let msg = format!("conv2d_fused {label} CUDA vs CPU");
                            assert_tensor_allclose_tol(&got, &cpu, rtol, atol, &msg);
                        });
                    }

                    #[cfg(feature = "wgpu")]
                    if is_dtype_supported("wgpu", dtype) {
                        with_wgpu_backend(|client, device| {
                            let ops = operands(&client, &device, dtype, case);
                            let got = fused(&client, &ops, case, layout, act, true, true);
                            let msg = format!("conv2d_fused {label} WebGPU vs CPU");
                            assert_tensor_allclose_tol(&got, &cpu, rtol, atol, &msg);
                        });
                    }
                }
            }
        }
    }
}

/// Residual scale 1.0 takes the `add(residual, y)` branch on every backend.
#[test]
fn test_conv2d_fused_backend_parity_unit_residual_scale() {
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        for case in CASES {
            let cpu_ops = operands(&cpu_client, &cpu_device, dtype, case);
            let (rtol, atol) = gemm_long_k_tolerance(dtype, case.contraction(), 1.0);
            for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
                for act in ACTS {
                    let cpu =
                        fused_scaled(&cpu_client, &cpu_ops, case, layout, act, true, Some(1.0));
                    let label = format!("{} {layout:?} {act:?} scale=1.0 [{dtype:?}]", case.name);

                    #[cfg(feature = "cuda")]
                    if is_dtype_supported("cuda", dtype) {
                        with_cuda_backend(|client, device| {
                            let ops = operands(&client, &device, dtype, case);
                            let got =
                                fused_scaled(&client, &ops, case, layout, act, true, Some(1.0));
                            let msg = format!("conv2d_fused {label} CUDA vs CPU");
                            assert_tensor_allclose_tol(&got, &cpu, rtol, atol, &msg);
                        });
                    }

                    #[cfg(feature = "wgpu")]
                    if is_dtype_supported("wgpu", dtype) {
                        with_wgpu_backend(|client, device| {
                            let ops = operands(&client, &device, dtype, case);
                            let got =
                                fused_scaled(&client, &ops, case, layout, act, true, Some(1.0));
                            let msg = format!("conv2d_fused {label} WebGPU vs CPU");
                            assert_tensor_allclose_tol(&got, &cpu, rtol, atol, &msg);
                        });
                    }
                }
            }
        }
    }
}
