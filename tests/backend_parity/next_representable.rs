//! Exact-bit parity using actual GPU clients and independent scalar references.

#[path = "../next_representable/fixtures.rs"]
mod fixtures;
#[path = "../next_representable/suite.rs"]
mod suite;

use numr::prelude::*;

#[test]
fn next_representable_cpu_matches_independent_reference() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    suite::f32_suite(&client, &device)?;
    suite::f64_suite(&client, &device)?;
    suite::unsupported(&client, &device, false)
}

#[cfg(feature = "cuda")]
#[test]
fn next_representable_cuda_matches_cpu_bits() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let cpu_device = CpuDevice::new();
    let cpu = CpuRuntime::default_client(&cpu_device);
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone())?;
    assert_eq!(
        suite::f32_suite(&client, &device)?,
        suite::f32_suite(&cpu, &cpu_device)?
    );
    assert_eq!(
        suite::f64_suite(&client, &device)?,
        suite::f64_suite(&cpu, &cpu_device)?
    );
    suite::unsupported(&client, &device, false)?;
    let foreign_device = CudaDevice::new(1);
    let foreign = Tensor::<CudaRuntime>::empty(&[2, 0, 3], DType::F32, &foreign_device)?;
    assert!(matches!(
        client.next_up(&foreign),
        Err(Error::DeviceMismatch)
    ));
    assert!(matches!(
        client.next_down(&foreign),
        Err(Error::DeviceMismatch)
    ));
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn next_representable_wgpu_matches_cpu_bits() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let cpu_device = CpuDevice::new();
    let cpu = CpuRuntime::default_client(&cpu_device);
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone())?;
    assert_eq!(
        suite::f32_suite(&client, &device)?,
        suite::f32_suite(&cpu, &cpu_device)?
    );
    suite::unsupported(&client, &device, true)?;
    let foreign_device = WgpuDevice::new(1);
    let foreign = Tensor::<WgpuRuntime>::empty(&[2, 0, 3], DType::F32, &foreign_device)?;
    assert!(matches!(
        client.next_up(&foreign),
        Err(Error::DeviceMismatch)
    ));
    assert!(matches!(
        client.next_down(&foreign),
        Err(Error::DeviceMismatch)
    ));
    Ok(())
}
