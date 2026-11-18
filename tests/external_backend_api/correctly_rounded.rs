//! `require_correctly_rounded_contracts` on runtimes that lack a contract.

use super::{MockAllocator, MockDevice, MockRuntime};
use numr::dtype::DType;
use numr::error;
use numr::prelude::*;
use numr::runtime::cpu::CpuRuntime;

/// A runtime that declares IEEE basic arithmetic but leaves fused multiply-add backend defined.
#[derive(Clone)]
struct BasicOnlyRuntime;

#[derive(Clone)]
struct BasicOnlyClient;

impl Runtime for BasicOnlyRuntime {
    type Device = MockDevice;
    type Client = BasicOnlyClient;
    type Allocator = MockAllocator;
    type Graph = numr::runtime::NoOpGraph;
    type RawHandle = ();
    type DType = DType;

    fn name() -> &'static str {
        "basic-only"
    }

    fn float_arithmetic_contract(_dtype: DType) -> FloatArithmeticContract {
        FloatArithmeticContract::Ieee754Basic
    }

    fn capture_graph_into<F>(
        client: &Self::Client,
        inputs: &[&Tensor<Self>],
        outputs: &[&Tensor<Self>],
        f: F,
    ) -> error::Result<CapturedGraph<Self>>
    where
        F: FnOnce(&Self::Client) -> error::Result<()>,
    {
        f(client)?;
        Ok(CapturedGraph::new(
            NoOpGraph,
            inputs.iter().map(|t| (*t).clone()).collect(),
            outputs.iter().map(|t| (*t).clone()).collect(),
        ))
    }

    fn allocate(_size_bytes: usize, _device: &MockDevice) -> error::Result<u64> {
        Ok(0)
    }

    fn deallocate(_ptr: u64, _size_bytes: usize, _device: &MockDevice) {}

    fn copy_to_device(_src: &[u8], _dst: u64, _device: &MockDevice) -> error::Result<()> {
        Ok(())
    }

    fn copy_from_device(_src: u64, _dst: &mut [u8], _device: &MockDevice) -> error::Result<()> {
        Ok(())
    }

    fn copy_within_device(
        _src: u64,
        _dst: u64,
        _size_bytes: usize,
        _device: &MockDevice,
    ) -> error::Result<()> {
        Ok(())
    }

    fn copy_strided(
        _src_handle: u64,
        _src_byte_offset: usize,
        _dst_handle: u64,
        _shape: &[usize],
        _strides: &[isize],
        _elem_size: usize,
        _device: &MockDevice,
    ) -> error::Result<()> {
        Ok(())
    }

    fn default_device() -> MockDevice {
        MockDevice
    }

    fn default_client(_device: &MockDevice) -> BasicOnlyClient {
        BasicOnlyClient
    }

    fn raw_handle(_client: &BasicOnlyClient) -> &() {
        &()
    }
}

impl RuntimeClient<BasicOnlyRuntime> for BasicOnlyClient {
    fn device(&self) -> &MockDevice {
        &MockDevice
    }

    fn synchronize(&self) {}

    fn allocator(&self) -> &MockAllocator {
        &MockAllocator
    }
}

#[test]
fn correctly_rounded_rejects_runtime_without_basic_arithmetic() {
    let err = numr::ops::require_correctly_rounded_contracts::<MockRuntime>("cr_exp")
        .expect_err("backend-defined arithmetic must be rejected");
    match &err {
        Error::BackendLimitation {
            backend, operation, ..
        } => {
            assert_eq!(*backend, "mock");
            assert_eq!(*operation, "cr_exp");
        }
        other => panic!("unexpected error {other:?}"),
    }
    assert!(err.to_string().contains("IEEE 754 basic"), "{err}");
}

#[test]
fn correctly_rounded_rejects_runtime_without_fused_mul_add() {
    let err = numr::ops::require_correctly_rounded_contracts::<BasicOnlyRuntime>("cr_log1p")
        .expect_err("backend-defined fused multiply-add must be rejected");
    let message = err.to_string();
    assert!(matches!(err, Error::InvalidArgument { .. }), "{err:?}");
    assert!(message.contains("cr_log1p"), "{message}");
    assert!(message.contains("fused multiply-add for F64"), "{message}");
}

#[test]
fn correctly_rounded_accepts_cpu_contracts() {
    assert!(numr::ops::require_correctly_rounded_contracts::<CpuRuntime>("cr_pow_scalar").is_ok());
}
