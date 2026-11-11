use super::cases;

#[test]
fn cpu_index_put_owns_output() {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    cases::contiguous::<CpuRuntime, _, _>(&client, &device, |v| v as f32);
    cases::contiguous::<CpuRuntime, _, _>(&client, &device, |v| v as f64);
    cases::contiguous::<CpuRuntime, _, _>(&client, &device, |v| v);
    cases::views_and_aliases::<CpuRuntime, _, _>(&client, &device, |v| v as f32);
    cases::views_and_aliases::<CpuRuntime, _, _>(&client, &device, |v| v as f64);
    cases::views_and_aliases::<CpuRuntime, _, _>(&client, &device, |v| v);
    cases::negative_strides::<CpuRuntime, _, _>(&client, &device, |v| v as f32);
    cases::negative_strides::<CpuRuntime, _, _>(&client, &device, |v| v as f64);
    cases::negative_strides::<CpuRuntime, _, _>(&client, &device, |v| v);
    integer_payloads::<CpuRuntime, _>(&client, &device);
    floating_payloads::<CpuRuntime, _>(&client, &device);
}

fn integer_payloads<R, C>(client: &C, device: &R::Device)
where
    R: numr::runtime::Runtime<DType = numr::dtype::DType>,
    C: numr::ops::IndexingOps<R>,
{
    cases::payloads::<R, _, _>(
        client,
        device,
        &[i64::MIN + 1, 0, (1i64 << 54) + 1, -1],
        &[i64::MAX, -(1i64 << 54) - 1],
    );
}

fn floating_payloads<R, C>(client: &C, device: &R::Device)
where
    R: numr::runtime::Runtime<DType = numr::dtype::DType>,
    C: numr::ops::IndexingOps<R>,
{
    cases::payloads::<R, _, _>(
        client,
        device,
        &[f32::from_bits(0x7fc0_1234), 1.0, -0.0, 2.0],
        &[0.0f32, f32::from_bits(0xffc0_5678)],
    );
    cases::payloads::<R, _, _>(
        client,
        device,
        &[f64::from_bits(0x7ff8_0000_0000_1234), 1.0, -0.0, 2.0],
        &[0.0f64, f64::from_bits(0xfff8_0000_0000_5678)],
    );
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_contiguous_index_put_owns_output() {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone()).expect("actual CUDA device required");
    cases::contiguous::<CudaRuntime, _, _>(&client, &device, |v| v as f32);
    cases::contiguous::<CudaRuntime, _, _>(&client, &device, |v| v as f64);
    cases::contiguous::<CudaRuntime, _, _>(&client, &device, |v| v);
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_index_put_views_aliases_and_payloads() {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone()).expect("actual CUDA device required");
    cases::views_and_aliases::<CudaRuntime, _, _>(&client, &device, |v| v as f32);
    cases::views_and_aliases::<CudaRuntime, _, _>(&client, &device, |v| v as f64);
    cases::views_and_aliases::<CudaRuntime, _, _>(&client, &device, |v| v);
    cases::negative_strides::<CudaRuntime, _, _>(&client, &device, |v| v as f32);
    cases::negative_strides::<CudaRuntime, _, _>(&client, &device, |v| v as f64);
    cases::negative_strides::<CudaRuntime, _, _>(&client, &device, |v| v);
    integer_payloads::<CudaRuntime, _>(&client, &device);
    floating_payloads::<CudaRuntime, _>(&client, &device);
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_index_put_owns_output() {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone()).expect("actual WebGPU device required");
    cases::contiguous::<WgpuRuntime, _, _>(&client, &device, |v| v as f32);
    cases::views_and_aliases::<WgpuRuntime, _, _>(&client, &device, |v| v as f32);
}
