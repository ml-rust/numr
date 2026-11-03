// Backend parity for `FusedConvOps::conv2d_fused` and `conv2d_fused_into`.
//
// CPU correctness (`reference`): the fused op must match `conv2d` + activation
// + residual composed by hand, for every activation, both layouts, with and
// without bias and residual, across stride, padding, dilation and group cases.
// NHWC runs are permuted to NCHW before the comparison.
//
// `_into` (`into`): the RRDB dense-block pattern writes each conv into the next
// channel slice of one buffer while reading the growing prefix of the same
// buffer. Channels outside the written slice must keep their values exactly.
//
// CUDA and WebGPU (`parity`) must match the CPU result. Their conv runs as
// im2col + GEMM with a different accumulation order, so the tolerance is
// accumulation-aware.
//
// This file holds only the declarations and the helpers shared by more than
// one test file.

mod into;
mod parity;
mod reference;

use numr::dtype::DType;
use numr::ops::{
    Conv2dConfig, ConvActivation, ConvLayout, ConvOps, ConvResidual, FusedConvOps, PaddingMode,
    TypeConversionOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;

const NCHW_TO_NHWC: [usize; 4] = [0, 2, 3, 1];
const NHWC_TO_NCHW: [usize; 4] = [0, 3, 1, 2];
/// OIHW to OHWI is the same axis move as NCHW to NHWC.
const OIHW_TO_OHWI: [usize; 4] = NCHW_TO_NHWC;

const BATCH: usize = 2;
const C_IN: usize = 4;
const HEIGHT: usize = 6;
const WIDTH: usize = 7;
const C_OUT: usize = 4;
const KERNEL: usize = 3;
const LEAKY_SLOPE: f64 = 0.2;
const RESIDUAL_SCALE: f64 = 0.2;

/// Values in `[-0.5, 0.5)` that vary with the flat index.
fn pattern(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 37 + salt) % 97) as f64 / 97.0 - 0.5)
        .collect()
}

/// Positive PReLU slopes in `[0.1, 0.5)`.
fn slopes(n: usize, salt: usize) -> Vec<f64> {
    pattern(n, salt)
        .into_iter()
        .map(|v| v * 0.4 + 0.3)
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Act {
    None,
    Relu,
    Leaky,
    PreluPerChannel,
    PreluShared,
}

const ACTS: [Act; 5] = [
    Act::None,
    Act::Relu,
    Act::Leaky,
    Act::PreluPerChannel,
    Act::PreluShared,
];

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
}

const CASES: [Case; 4] = [
    Case {
        name: "stride2_valid",
        stride: (2, 2),
        padding: PaddingMode::Valid,
        dilation: (1, 1),
        groups: 1,
    },
    Case {
        name: "same",
        stride: (1, 1),
        padding: PaddingMode::Same,
        dilation: (1, 1),
        groups: 1,
    },
    Case {
        name: "dilation2_same",
        stride: (1, 1),
        padding: PaddingMode::Same,
        dilation: (2, 2),
        groups: 1,
    },
    Case {
        name: "groups2_pad1",
        stride: (1, 1),
        padding: PaddingMode::Custom(1, 1, 1, 1),
        dilation: (1, 1),
        groups: 2,
    },
];

impl Case {
    fn config(self, layout: ConvLayout) -> Conv2dConfig {
        Conv2dConfig::new(layout)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_dilation(self.dilation)
            .with_groups(self.groups)
    }

    /// GEMM contraction length of one output element.
    fn contraction(self) -> usize {
        C_IN / self.groups * KERNEL * KERNEL
    }
}

/// Every operand of one fused call, built in NCHW / OIHW order.
struct Operands<R: Runtime> {
    x: Tensor<R>,
    w: Tensor<R>,
    bias: Tensor<R>,
    slope_c: Tensor<R>,
    slope_1: Tensor<R>,
    residual: Tensor<R>,
}

fn operands<R, C>(client: &C, device: &R::Device, dtype: DType, case: Case) -> Operands<R>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + TypeConversionOps<R>,
{
    let make = |data: Vec<f64>, shape: &[usize]| {
        tensor_from_f64(&data, shape, dtype, device, client)
            .unwrap_or_else(|e| panic!("tensor_from_f64 {shape:?} [{dtype:?}]: {e}"))
    };
    let w_shape = [C_OUT, C_IN / case.groups, KERNEL, KERNEL];
    let x = make(
        pattern(BATCH * C_IN * HEIGHT * WIDTH, 1),
        &[BATCH, C_IN, HEIGHT, WIDTH],
    );
    let w = make(pattern(w_shape.iter().product(), 2), &w_shape);
    // The plain conv gives the output shape the residual must have.
    let out_shape = client
        .conv2d(
            &x,
            &w,
            None,
            case.stride,
            case.padding,
            case.dilation,
            case.groups,
        )
        .unwrap_or_else(|e| panic!("conv2d {} [{dtype:?}]: {e}", case.name))
        .shape()
        .to_vec();
    Operands {
        bias: make(pattern(C_OUT, 3), &[C_OUT]),
        slope_c: make(slopes(C_OUT, 4), &[C_OUT]),
        slope_1: make(slopes(1, 5), &[1]),
        residual: make(pattern(out_shape.iter().product(), 6), &out_shape),
        x,
        w,
    }
}

fn to_layout<R: Runtime<DType = DType>>(t: &Tensor<R>, perm: [usize; 4]) -> Tensor<R> {
    t.permute(&perm).unwrap().contiguous().unwrap()
}

/// `conv2d_fused` in `layout`, returned in NCHW order.
fn fused<R, C>(
    client: &C,
    ops: &Operands<R>,
    case: Case,
    layout: ConvLayout,
    act: Act,
    with_bias: bool,
    with_residual: bool,
) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: FusedConvOps<R>,
{
    fused_scaled(
        client,
        ops,
        case,
        layout,
        act,
        with_bias,
        with_residual.then_some(RESIDUAL_SCALE),
    )
}

/// `conv2d_fused` in `layout` with a residual of the given scale (`None` means
/// no residual), returned in NCHW order.
fn fused_scaled<R, C>(
    client: &C,
    ops: &Operands<R>,
    case: Case,
    layout: ConvLayout,
    act: Act,
    with_bias: bool,
    residual_scale: Option<f64>,
) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: FusedConvOps<R>,
{
    let (x, w, residual) = match layout {
        ConvLayout::Nchw => (ops.x.clone(), ops.w.clone(), ops.residual.clone()),
        ConvLayout::Nhwc => (
            to_layout(&ops.x, NCHW_TO_NHWC),
            to_layout(&ops.w, OIHW_TO_OHWI),
            to_layout(&ops.residual, NCHW_TO_NHWC),
        ),
    };
    let activation = match act {
        Act::None => ConvActivation::None,
        Act::Relu => ConvActivation::Relu,
        Act::Leaky => ConvActivation::LeakyRelu(LEAKY_SLOPE),
        Act::PreluPerChannel => ConvActivation::Prelu(&ops.slope_c),
        Act::PreluShared => ConvActivation::Prelu(&ops.slope_1),
    };
    let residual = residual_scale.map(|scale| ConvResidual {
        tensor: &residual,
        scale,
    });
    let out = client
        .conv2d_fused(
            &x,
            &w,
            with_bias.then_some(&ops.bias),
            &case.config(layout),
            activation,
            residual,
        )
        .unwrap_or_else(|e| panic!("conv2d_fused {} {layout:?} {act:?}: {e}", case.name));
    assert!(
        out.is_contiguous(),
        "conv2d_fused must return a contiguous tensor"
    );
    match layout {
        ConvLayout::Nchw => out,
        ConvLayout::Nhwc => to_layout(&out, NHWC_TO_NCHW),
    }
}
