// Backend parity tests migrated from tests/index_ops/masked.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use numr::ops::IndexingOps;
#[cfg(feature = "cuda")]
use numr::runtime::Runtime;
#[cfg(feature = "cuda")]
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

#[test]
fn test_masked_ops_parity() {
    #[cfg(feature = "cuda")]
    let cpu_device = CpuDevice::new();
    #[cfg(feature = "cuda")]
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    #[cfg(feature = "cuda")]
    let a_cpu =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu_device);
    #[cfg(feature = "cuda")]
    let mask_row_cpu = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1], &[1, 3], &cpu_device);
    #[cfg(feature = "cuda")]
    let cpu_select_row: Vec<f32> = cpu_client
        .masked_select(&a_cpu, &mask_row_cpu)
        .unwrap()
        .to_vec();
    #[cfg(feature = "cuda")]
    let cpu_fill_row: Vec<f32> = cpu_client
        .masked_fill(&a_cpu, &mask_row_cpu, -1.0)
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &cuda_device,
        );
        let mask_row = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1u8, 0, 1],
            &[1, 3],
            &cuda_device,
        );
        let select_row: Vec<f32> = cuda_client.masked_select(&a, &mask_row).unwrap().to_vec();
        assert_eq!(cpu_select_row, select_row);
        let fill_row: Vec<f32> = cuda_client
            .masked_fill(&a, &mask_row, -1.0)
            .unwrap()
            .to_vec();
        assert_eq!(cpu_fill_row, fill_row);

        let mask_col = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1u8, 0],
            &[2, 1],
            &cuda_device,
        );
        let select_col: Vec<f32> = cuda_client.masked_select(&a, &mask_col).unwrap().to_vec();
        assert_eq!(select_col, vec![1.0, 2.0, 3.0]);
        let fill_col: Vec<f32> = cuda_client
            .masked_fill(&a, &mask_col, 99.0)
            .unwrap()
            .to_vec();
        assert_eq!(fill_col, vec![99.0, 99.0, 99.0, 4.0, 5.0, 6.0]);

        let a3 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
            &cuda_device,
        );
        let m3 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1u8, 0],
            &[1, 2, 1],
            &cuda_device,
        );
        let d3: Vec<f32> = cuda_client.masked_select(&a3, &m3).unwrap().to_vec();
        assert_eq!(d3, vec![1.0, 2.0, 5.0, 6.0]);

        let a64 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0],
            &[2, 2],
            &cuda_device,
        );
        let m64 = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1u8, 0],
            &[2, 1],
            &cuda_device,
        );
        let d64: Vec<f64> = cuda_client
            .masked_fill(&a64, &m64, -999.0)
            .unwrap()
            .to_vec();
        assert_eq!(d64, vec![-999.0, -999.0, 3.0, 4.0]);
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let a = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &wgpu_device,
        );
        let mask = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &[1u32, 0, 1, 0, 0, 1, 0, 1],
            &[2, 4],
            &wgpu_device,
        );

        let selected: Vec<f32> = wgpu_client.masked_select(&a, &mask).unwrap().to_vec();
        assert_eq!(selected, vec![1.0, 3.0, 6.0, 8.0]);

        let filled: Vec<f32> = wgpu_client.masked_fill(&a, &mask, -1.0).unwrap().to_vec();
        assert_eq!(filled, vec![-1.0, 2.0, -1.0, 4.0, 5.0, -1.0, 7.0, -1.0]);
    });
}
