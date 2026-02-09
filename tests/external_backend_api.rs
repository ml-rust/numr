//! Compile test: simulates an external crate implementing numr traits.
//!
//! If this test compiles, our API is implementable by downstream crates.
//! Extension traits with default impls should not require method implementations.

use numr::error;
use numr::prelude::*;
use numr::runtime::Allocator;

// =============================================================================
// Mock backend types
// =============================================================================

#[derive(Clone)]
struct MockDevice;

impl Device for MockDevice {
    fn id(&self) -> usize {
        0
    }
}

#[derive(Clone)]
struct MockAllocator;

impl Allocator for MockAllocator {
    fn allocate(&self, _size_bytes: usize) -> error::Result<u64> {
        Ok(0)
    }

    fn deallocate(&self, _ptr: u64, _size_bytes: usize) {}
}

#[derive(Clone)]
struct MockClient;

#[derive(Clone)]
struct MockRuntime;

impl Runtime for MockRuntime {
    type Device = MockDevice;
    type Client = MockClient;
    type Allocator = MockAllocator;
    type RawHandle = ();

    fn name() -> &'static str {
        "mock"
    }

    fn allocate(_size_bytes: usize, _device: &Self::Device) -> error::Result<u64> {
        Ok(0)
    }

    fn deallocate(_ptr: u64, _size_bytes: usize, _device: &Self::Device) {}

    fn copy_to_device(_src: &[u8], _dst: u64, _device: &Self::Device) -> error::Result<()> {
        Ok(())
    }

    fn copy_from_device(_src: u64, _dst: &mut [u8], _device: &Self::Device) -> error::Result<()> {
        Ok(())
    }

    fn copy_within_device(
        _src: u64,
        _dst: u64,
        _size_bytes: usize,
        _device: &Self::Device,
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
        _device: &Self::Device,
    ) -> error::Result<()> {
        Ok(())
    }

    fn default_device() -> Self::Device {
        MockDevice
    }

    fn default_client(_device: &Self::Device) -> Self::Client {
        MockClient
    }

    fn raw_handle(_client: &Self::Client) -> &Self::RawHandle {
        &()
    }
}

impl RuntimeClient<MockRuntime> for MockClient {
    fn device(&self) -> &MockDevice {
        &MockDevice
    }

    fn synchronize(&self) {}

    fn allocator(&self) -> &MockAllocator {
        &MockAllocator
    }
}

// =============================================================================
// Extension traits: implementing with ONLY defaults should compile.
// This is the key test - no method bodies needed for extension traits.
// =============================================================================

impl ActivationOps<MockRuntime> for MockClient {}
impl NormalizationOps<MockRuntime> for MockClient {}
impl ConvOps<MockRuntime> for MockClient {}
impl StatisticalOps<MockRuntime> for MockClient {}
impl SortingOps<MockRuntime> for MockClient {}
impl IndexingOps<MockRuntime> for MockClient {}
impl RandomOps<MockRuntime> for MockClient {}
impl AdvancedRandomOps<MockRuntime> for MockClient {}
impl QuasiRandomOps<MockRuntime> for MockClient {}
impl MultivariateRandomOps<MockRuntime> for MockClient {}
impl CumulativeOps<MockRuntime> for MockClient {}
impl ComplexOps<MockRuntime> for MockClient {}
impl DistanceOps<MockRuntime> for MockClient {}
impl LogicalOps<MockRuntime> for MockClient {}
impl LinalgOps<MockRuntime> for MockClient {}
impl ConditionalOps<MockRuntime> for MockClient {}
impl TypeConversionOps<MockRuntime> for MockClient {}
impl UtilityOps<MockRuntime> for MockClient {}

// Algorithm traits with defaults
use numr::algorithm::SpecialFunctions;
use numr::algorithm::fft::FftAlgorithms;
use numr::algorithm::linalg::{
    LinearAlgebraAlgorithms, MatrixFunctionsAlgorithms, TensorDecomposeAlgorithms,
};
use numr::algorithm::polynomial::PolynomialAlgorithms;

impl FftAlgorithms<MockRuntime> for MockClient {}
impl MatrixFunctionsAlgorithms<MockRuntime> for MockClient {}
impl LinearAlgebraAlgorithms<MockRuntime> for MockClient {}
impl TensorDecomposeAlgorithms<MockRuntime> for MockClient {}
impl SpecialFunctions<MockRuntime> for MockClient {}
impl PolynomialAlgorithms<MockRuntime> for MockClient {}

// =============================================================================
// This test just needs to compile - the fact that it compiles proves
// the external backend API is implementable.
// =============================================================================

#[test]
fn external_backend_compiles() {
    // If we got here, all traits compiled with empty impls
    let _device = MockRuntime::default_device();
    let _client = MockRuntime::default_client(&_device);
}
