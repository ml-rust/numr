// Shared driver for the autograd backend parity tests.
//
// A case names its leaf inputs and a `GraphOp` that maps the leaves to one
// output. The driver runs the op on every enabled backend, then runs
// `backward(sum(op(x) * w))` with a fixed, non-uniform weight `w`. CPU is the
// reference. CUDA and WebGPU must match its forward value and every leaf
// gradient. All tensors are F32, the one float dtype WebGPU carries.

use numr::autograd::{Var, backward, var_mul, var_sum};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{CompareOps, EinsumOps, ScalarOps, SpecialFunctions, TensorOps};
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{assert_tensor_allclose_tol, create_cpu_client};

/// Every op trait the cases call, on one client type.
pub trait ParityClient<R: Runtime>:
    RuntimeClient<R>
    + TensorOps<R>
    + ScalarOps<R>
    + CompareOps<R>
    + SpecialFunctions<R>
    + EinsumOps<R>
    + 'static
{
}

impl<R: Runtime, C> ParityClient<R> for C where
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + SpecialFunctions<R>
        + EinsumOps<R>
        + 'static
{
}

/// The backend a case runs on. It picks the index and mask carriers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

/// Client, device and backend for one run.
pub struct Ctx<'a, R: Runtime> {
    pub client: &'a R::Client,
    pub device: &'a R::Device,
    pub backend: Backend,
}

impl<R: Runtime<DType = DType>> Ctx<'_, R> {
    /// Failure label for this backend against the CPU reference.
    pub fn label(&self, name: &str) -> String {
        format!("{name} {:?} vs CPU", self.backend)
    }

    fn is_wgpu(&self) -> bool {
        #[cfg(feature = "wgpu")]
        {
            self.backend == Backend::Wgpu
        }
        #[cfg(not(feature = "wgpu"))]
        {
            false
        }
    }

    /// Index tensor in the backend's widest index dtype.
    ///
    /// CPU and CUDA get I64, the dtype the autograd backward passes store.
    /// WebGPU has no 64-bit integer type, so it gets I32.
    pub fn index(&self, values: &[i64], shape: &[usize]) -> Result<Tensor<R>> {
        if self.is_wgpu() {
            self.index_i32(values, shape)
        } else {
            Tensor::from_slice(values, shape, self.device)
        }
    }

    /// Index tensor in I32 on every backend.
    pub fn index_i32(&self, values: &[i64], shape: &[usize]) -> Result<Tensor<R>> {
        let narrow: Vec<i32> = values.iter().map(|&v| v as i32).collect();
        Tensor::from_slice(&narrow, shape, self.device)
    }

    /// Boolean mask in the carrier `where_cond` takes on this backend.
    ///
    /// CPU and CUDA take a byte. WebGPU takes a 32-bit word.
    pub fn mask(&self, values: &[bool], shape: &[usize]) -> Result<Tensor<R>> {
        if self.is_wgpu() {
            let words: Vec<u32> = values.iter().map(|&v| u32::from(v)).collect();
            Tensor::from_slice(&words, shape, self.device)
        } else {
            let bytes: Vec<u8> = values.iter().map(|&v| u8::from(v)).collect();
            Tensor::from_slice(&bytes, shape, self.device)
        }
    }

    /// F32 tensor from f64 host data.
    pub fn f32(&self, data: &[f64], shape: &[usize]) -> Result<Tensor<R>>
    where
        R::Client: TensorOps<R>,
    {
        tensor_from_f64(data, shape, DType::F32, self.device, self.client)
    }
}

/// An op under test: maps leaf `Var`s to one output `Var`.
pub trait GraphOp {
    fn build<R>(&self, leaves: &[Var<R>], ctx: &Ctx<'_, R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ParityClient<R>;
}

/// Define a unit struct `$name` that implements `GraphOp` with `$body`.
///
/// Doc comments before `$name` land on the struct.
macro_rules! graph_op {
    ($(#[$meta:meta])* $name:ident, |$v:ident, $ctx:ident| $body:expr) => {
        $(#[$meta])*
        struct $name;

        impl $crate::backend_parity::autograd_ops::harness::GraphOp for $name {
            fn build<R>(
                &self,
                $v: &[numr::autograd::Var<R>],
                $ctx: &$crate::backend_parity::autograd_ops::harness::Ctx<'_, R>,
            ) -> numr::error::Result<numr::autograd::Var<R>>
            where
                R: numr::runtime::Runtime<DType = numr::dtype::DType>,
                R::Client: $crate::backend_parity::autograd_ops::harness::ParityClient<R>,
            {
                $body
            }
        }
    };
}
pub(crate) use graph_op;

/// One trainable leaf, as host data.
pub struct Leaf {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

pub fn leaf(data: &[f64], shape: &[usize]) -> Leaf {
    let numel: usize = shape.iter().product();
    assert_eq!(data.len(), numel, "leaf data does not fill shape {shape:?}");
    Leaf {
        data: data.to_vec(),
        shape: shape.to_vec(),
    }
}

/// `(rtol, atol)` for one comparison: `|a - b| <= atol + rtol * |b|`.
#[derive(Copy, Clone, Debug)]
pub struct Tol {
    pub rtol: f64,
    pub atol: f64,
}

/// Elementwise and indexing ops: one rounding per value.
pub const TOL: Tol = Tol {
    rtol: 1e-4,
    atol: 1e-5,
};

/// Reductions, products and contractions: summation order differs per backend.
pub const REDUCE_TOL: Tol = Tol {
    rtol: 5e-4,
    atol: 5e-5,
};

/// What WebGPU must do for a case.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OnWgpu {
    /// The case runs and matches CPU.
    Run,
    /// The case matches CPU, or fails with a clean unsupported-class `Err`.
    /// A panic or any other error still fails the test.
    AllowUnsupported,
}

/// Labelled, contiguous tensors that one run produces on one backend.
pub type Probed<R> = Vec<(String, Tensor<R>)>;

/// A whole scenario run per backend: every tensor it returns must match CPU.
///
/// `GraphOp` cases run through `GraphCase`. Scenarios with their own graph,
/// such as double backward, grad mode and checkpointing, implement this directly.
pub trait Probe {
    fn run<R>(&self, ctx: &Ctx<'_, R>) -> Result<Probed<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ParityClient<R>;
}

/// A `GraphOp` with its leaves: probes the forward value and every leaf gradient.
pub struct GraphCase<'a, O> {
    pub op: &'a O,
    pub leaves: &'a [Leaf],
}

impl<O: GraphOp> Probe for GraphCase<'_, O> {
    fn run<R>(&self, ctx: &Ctx<'_, R>) -> Result<Probed<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ParityClient<R>,
    {
        let vars = leaf_vars(self.leaves, ctx)?;
        let out = self.op.build(&vars, ctx)?;
        let loss = weighted_loss(&out, ctx)?;
        let grads = backward(&loss, ctx.client)?;
        let mut probed = vec![("forward".to_string(), out.tensor().contiguous()?)];
        for (i, v) in vars.iter().enumerate() {
            let grad = grads.get(v.id()).ok_or(Error::MissingGradient)?;
            probed.push((format!("grad[{i}]"), grad.contiguous()?));
        }
        Ok(probed)
    }
}

/// Fixed weight in `[-0.9, 1.1]` with no zero, so every output element
/// carries a distinct, nonzero upstream gradient.
pub fn weight_data(numel: usize) -> Vec<f64> {
    (0..numel)
        .map(|i| ((i * 5 + 3) % 9) as f64 * 0.25 - 0.9)
        .collect()
}

/// F32 leaf `Var`s that require grad.
pub fn leaf_vars<R>(leaves: &[Leaf], ctx: &Ctx<'_, R>) -> Result<Vec<Var<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: ParityClient<R>,
{
    leaves
        .iter()
        .map(|l| Ok(Var::new(ctx.f32(&l.data, &l.shape)?, true)))
        .collect()
}

/// `sum(out * w)` with the fixed weight sized to `out`.
pub fn weighted_loss<R>(out: &Var<R>, ctx: &Ctx<'_, R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ParityClient<R>,
{
    let w = Var::new(ctx.f32(&weight_data(out.numel()), out.shape())?, false);
    var_sum(&var_mul(out, &w, ctx.client)?, &[], false, ctx.client)
}

/// Assert shape, dtype and values of `actual` against `expected`.
///
/// The two tensors can live on different backends or on the same one.
pub fn assert_close<R1: Runtime<DType = DType>, R2: Runtime<DType = DType>>(
    actual: &Tensor<R1>,
    expected: &Tensor<R2>,
    tol: Tol,
    label: &str,
) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape differs from CPU"
    );
    assert_tensor_allclose_tol(actual, expected, tol.rtol, tol.atol, label);
}

/// Assert every probed device tensor against its CPU counterpart.
pub fn assert_probed<R: Runtime<DType = DType>>(
    actual: &Probed<R>,
    expected: &Probed<CpuRuntime>,
    tol: Tol,
    label: &str,
) {
    let actual_names: Vec<&str> = actual.iter().map(|(n, _)| n.as_str()).collect();
    let expected_names: Vec<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        actual_names, expected_names,
        "{label}: probed tensors differ from CPU"
    );
    for ((name, a), (_, e)) in actual.iter().zip(expected) {
        assert_close(a, e, tol, &format!("{label} {name}"));
    }
}

/// True for the error classes a backend returns for a missing kernel or dtype.
#[cfg(feature = "wgpu")]
pub fn is_unsupported(err: &Error) -> bool {
    matches!(
        err,
        Error::UnsupportedDType { .. }
            | Error::NotImplemented { .. }
            | Error::BackendLimitation { .. }
    )
}

/// Run one `GraphOp` case on CPU, then on every enabled backend, and compare
/// the forward value and every leaf gradient.
pub fn check_parity<O: GraphOp>(name: &str, op: &O, leaves: &[Leaf], tol: Tol, on_wgpu: OnWgpu) {
    check_probe(name, &GraphCase { op, leaves }, tol, on_wgpu);
}

/// Run `probe` on CPU, then on every enabled backend, and compare.
pub fn check_probe<P: Probe>(name: &str, probe: &P, tol: Tol, on_wgpu: OnWgpu) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_ctx = Ctx::<CpuRuntime> {
        client: &cpu_client,
        device: &cpu_device,
        backend: Backend::Cpu,
    };
    let cpu = probe
        .run(&cpu_ctx)
        .unwrap_or_else(|e| panic!("{name}: CPU reference failed: {e}"));

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let ctx = Ctx::<numr::runtime::cuda::CudaRuntime> {
            client: &client,
            device: &device,
            backend: Backend::Cuda,
        };
        let dev = probe
            .run(&ctx)
            .unwrap_or_else(|e| panic!("{name}: CUDA run failed: {e}"));
        assert_probed(&dev, &cpu, tol, &ctx.label(name));
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let ctx = Ctx::<numr::runtime::wgpu::WgpuRuntime> {
            client: &client,
            device: &device,
            backend: Backend::Wgpu,
        };
        match probe.run(&ctx) {
            Ok(dev) => assert_probed(&dev, &cpu, tol, &ctx.label(name)),
            Err(e) if on_wgpu == OnWgpu::AllowUnsupported && is_unsupported(&e) => {
                println!("{name}: WGPU unsupported, value check skipped: {e}");
            }
            Err(e) => panic!("{name}: WGPU run failed: {e}"),
        }
    });

    #[cfg(not(feature = "wgpu"))]
    let _ = on_wgpu;
}
