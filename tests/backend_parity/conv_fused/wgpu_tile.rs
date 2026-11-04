// WebGPU tile kernels of `conv2d` and `conv2d_fused` against the CPU result.
//
// The shapes cross each kernel edge: `C_in` a multiple of 4 or not, `C_out` a
// multiple of 4 or not, output sizes past whole tiles, and tiles past the
// output edge. Every call compares in NCHW order.

use numr::dtype::DType;
use numr::ops::{
    Conv2dConfig, ConvActivation, ConvLayout, ConvOps, ConvResidual, FusedConvOps, PaddingMode,
    TypeConversionOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::{LEAKY_SLOPE, NHWC_TO_NCHW, OIHW_TO_OHWI, RESIDUAL_SCALE, pattern, slopes, to_layout};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{assert_tensor_allclose_tol, create_cpu_client, gemm_long_k_tolerance};

const DTYPE: DType = DType::F32;

#[derive(Clone, Copy)]
struct Shape {
    batch: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kernel: usize,
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
}

const SAME: PaddingMode = PaddingMode::Same;

const SHAPES: [Shape; 7] = [
    // SRVGG body, head and tail blocks.
    Shape {
        batch: 1,
        c_in: 64,
        h: 9,
        w: 18,
        c_out: 64,
        kernel: 3,
        stride: (1, 1),
        padding: SAME,
        dilation: (1, 1),
    },
    Shape {
        batch: 1,
        c_in: 3,
        h: 17,
        w: 33,
        c_out: 64,
        kernel: 3,
        stride: (1, 1),
        padding: SAME,
        dilation: (1, 1),
    },
    Shape {
        batch: 1,
        c_in: 64,
        h: 8,
        w: 16,
        c_out: 48,
        kernel: 3,
        stride: (1, 1),
        padding: SAME,
        dilation: (1, 1),
    },
    // C_out and sizes that leave a part block and a part tile.
    Shape {
        batch: 2,
        c_in: 8,
        h: 11,
        w: 13,
        c_out: 6,
        kernel: 3,
        stride: (1, 1),
        padding: SAME,
        dilation: (1, 1),
    },
    // Stride, valid padding, one output channel.
    Shape {
        batch: 2,
        c_in: 5,
        h: 12,
        w: 19,
        c_out: 1,
        kernel: 3,
        stride: (2, 3),
        padding: PaddingMode::Valid,
        dilation: (1, 1),
    },
    // Dilation and asymmetric padding in both kernels.
    Shape {
        batch: 1,
        c_in: 12,
        h: 10,
        w: 21,
        c_out: 7,
        kernel: 3,
        stride: (1, 2),
        padding: PaddingMode::Custom(2, 1, 3, 0),
        dilation: (2, 1),
    },
    // One pixel: every tap but the center falls outside the input.
    Shape {
        batch: 1,
        c_in: 4,
        h: 1,
        w: 1,
        c_out: 4,
        kernel: 3,
        stride: (1, 1),
        padding: SAME,
        dilation: (1, 1),
    },
];

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

impl Shape {
    fn weight_shape(self) -> [usize; 4] {
        [self.c_out, self.c_in, self.kernel, self.kernel]
    }

    fn config(self, layout: ConvLayout) -> Conv2dConfig {
        Conv2dConfig::new(layout)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_dilation(self.dilation)
    }
}

/// One run in NCHW order: every operand built on `device`, the layout applied,
/// `conv2d_fused` and `conv2d_fused_into` both checked.
fn run<R, C>(
    client: &C,
    device: &R::Device,
    shape: Shape,
    layout: ConvLayout,
    act: Act,
    with_bias: bool,
    with_residual: bool,
) -> (Tensor<R>, Tensor<R>)
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + FusedConvOps<R> + TypeConversionOps<R>,
{
    let make = |data: Vec<f64>, dims: &[usize]| {
        tensor_from_f64(&data, dims, DTYPE, device, client)
            .unwrap_or_else(|e| panic!("tensor_from_f64 {dims:?}: {e}"))
    };
    let input_dims = [shape.batch, shape.c_in, shape.h, shape.w];
    let weight_dims = shape.weight_shape();
    let x = make(pattern(input_dims.iter().product(), 11), &input_dims);
    let w = make(pattern(weight_dims.iter().product(), 12), &weight_dims);
    let bias = make(pattern(shape.c_out, 13), &[shape.c_out]);
    let slope_c = make(slopes(shape.c_out, 14), &[shape.c_out]);
    let slope_1 = make(slopes(1, 15), &[1]);

    // The plain conv gives the output shape.
    let plain = client
        .conv2d(&x, &w, None, shape.stride, shape.padding, shape.dilation, 1)
        .unwrap_or_else(|e| panic!("conv2d {input_dims:?}: {e}"));
    let out_dims = plain.shape().to_vec();
    let residual = make(pattern(out_dims.iter().product(), 16), &out_dims);

    let (x, w, residual) = match layout {
        ConvLayout::Nchw => (x, w, residual),
        ConvLayout::Nhwc => (
            to_layout(&x, OIHW_TO_OHWI),
            to_layout(&w, OIHW_TO_OHWI),
            to_layout(&residual, OIHW_TO_OHWI),
        ),
    };
    let activation = match act {
        Act::None => ConvActivation::None,
        Act::Relu => ConvActivation::Relu,
        Act::Leaky => ConvActivation::LeakyRelu(LEAKY_SLOPE),
        Act::PreluPerChannel => ConvActivation::Prelu(&slope_c),
        Act::PreluShared => ConvActivation::Prelu(&slope_1),
    };
    let residual = with_residual.then_some(ConvResidual {
        tensor: &residual,
        scale: RESIDUAL_SCALE,
    });
    let bias = with_bias.then_some(&bias);
    let config = shape.config(layout);
    let fused = client
        .conv2d_fused(&x, &w, bias, &config, activation, residual)
        .unwrap_or_else(|e| panic!("conv2d_fused {input_dims:?} {layout:?} {act:?}: {e}"));
    let into = Tensor::<R>::empty(fused.shape(), DTYPE, device)
        .unwrap_or_else(|e| panic!("empty {:?}: {e}", fused.shape()));
    client
        .conv2d_fused_into(&into, &x, &w, bias, &config, activation, residual)
        .unwrap_or_else(|e| panic!("conv2d_fused_into {input_dims:?} {layout:?} {act:?}: {e}"));
    let nchw = |t: Tensor<R>| match layout {
        ConvLayout::Nchw => t,
        ConvLayout::Nhwc => to_layout(&t, NHWC_TO_NCHW),
    };
    (nchw(fused), nchw(into))
}

fn tolerance(shape: Shape) -> (f64, f64) {
    gemm_long_k_tolerance(DTYPE, shape.c_in * shape.kernel * shape.kernel, 1.0)
}

#[test]
fn tile_kernels_match_cpu_for_every_shape_and_activation() {
    let (cpu_client, cpu_device) = create_cpu_client();
    for shape in SHAPES {
        let (rtol, atol) = tolerance(shape);
        for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
            for act in ACTS {
                let (cpu, _) = run(&cpu_client, &cpu_device, shape, layout, act, true, true);
                with_wgpu_backend(|client, device| {
                    let (fused, into) = run(&client, &device, shape, layout, act, true, true);
                    let label = format!(
                        "{}x{}x{} -> {} {layout:?} {act:?}",
                        shape.c_in, shape.h, shape.w, shape.c_out
                    );
                    let msg = format!("conv2d_fused {label} WebGPU vs CPU");
                    assert_tensor_allclose_tol(&fused, &cpu, rtol, atol, &msg);
                    let msg = format!("conv2d_fused_into {label} WebGPU vs CPU");
                    assert_tensor_allclose_tol(&into, &cpu, rtol, atol, &msg);
                });
            }
        }
    }
}

#[test]
fn tile_kernels_match_cpu_without_bias_or_residual() {
    let (cpu_client, cpu_device) = create_cpu_client();
    for shape in SHAPES {
        let (rtol, atol) = tolerance(shape);
        for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
            let (cpu, _) = run(
                &cpu_client,
                &cpu_device,
                shape,
                layout,
                Act::None,
                false,
                false,
            );
            with_wgpu_backend(|client, device| {
                let (fused, _) = run(&client, &device, shape, layout, Act::None, false, false);
                let msg = format!(
                    "conv2d_fused {}x{}x{} -> {} {layout:?} no bias WebGPU vs CPU",
                    shape.c_in, shape.h, shape.w, shape.c_out
                );
                assert_tensor_allclose_tol(&fused, &cpu, rtol, atol, &msg);
            });
        }
    }
}

#[test]
fn plain_conv2d_matches_cpu_for_every_shape() {
    let (cpu_client, cpu_device) = create_cpu_client();
    for shape in SHAPES {
        let (rtol, atol) = tolerance(shape);
        let input_dims = [shape.batch, shape.c_in, shape.h, shape.w];
        let weight_dims = shape.weight_shape();
        let x_data = pattern(input_dims.iter().product(), 21);
        let w_data = pattern(weight_dims.iter().product(), 22);
        let bias_data = pattern(shape.c_out, 23);
        let cpu = {
            let make = |data: &[f64], dims: &[usize]| {
                tensor_from_f64(data, dims, DTYPE, &cpu_device, &cpu_client).unwrap()
            };
            cpu_client
                .conv2d(
                    &make(&x_data, &input_dims),
                    &make(&w_data, &weight_dims),
                    Some(&make(&bias_data, &[shape.c_out])),
                    shape.stride,
                    shape.padding,
                    shape.dilation,
                    1,
                )
                .unwrap()
        };
        with_wgpu_backend(|client, device| {
            let make = |data: &[f64], dims: &[usize]| {
                tensor_from_f64(data, dims, DTYPE, &device, &client).unwrap()
            };
            let got = client
                .conv2d(
                    &make(&x_data, &input_dims),
                    &make(&w_data, &weight_dims),
                    Some(&make(&bias_data, &[shape.c_out])),
                    shape.stride,
                    shape.padding,
                    shape.dilation,
                    1,
                )
                .unwrap();
            let msg = format!("conv2d {input_dims:?} -> {} WebGPU vs CPU", shape.c_out);
            assert_tensor_allclose_tol(&got, &cpu, rtol, atol, &msg);
        });
    }
}
