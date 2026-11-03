// CPU reference correctness: the fused op against `conv2d` + activation +
// residual composed by hand, and NHWC against NCHW.

use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ConvLayout, ConvOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::{
    ACTS, Act, C_OUT, CASES, Case, LEAKY_SLOPE, Operands, RESIDUAL_SCALE, fused, fused_scaled,
    operands,
};
use crate::common::{DTypeDomain, assert_tensor_allclose, create_cpu_client, parity_dtypes};

/// `conv2d` + activation + residual composed by hand, NCHW.
fn composed<R, C>(
    client: &C,
    ops: &Operands<R>,
    case: Case,
    act: Act,
    with_bias: bool,
    with_residual: bool,
) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ActivationOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let bias = with_bias.then_some(&ops.bias);
    let y = client
        .conv2d(
            &ops.x,
            &ops.w,
            bias,
            case.stride,
            case.padding,
            case.dilation,
            case.groups,
        )
        .unwrap();
    let y = match act {
        Act::None => y,
        Act::Relu => client.relu(&y).unwrap(),
        Act::Leaky => client.leaky_relu(&y, LEAKY_SLOPE).unwrap(),
        Act::PreluPerChannel => {
            let slope = ops.slope_c.reshape(&[C_OUT, 1, 1]).unwrap();
            client.prelu(&y, &slope).unwrap()
        }
        Act::PreluShared => client.prelu(&y, &ops.slope_1).unwrap(),
    };
    if !with_residual {
        return y;
    }
    let scaled = client.mul_scalar(&y, RESIDUAL_SCALE).unwrap();
    client.add(&ops.residual, &scaled).unwrap()
}

fn cpu_matches_composition(act: Act) {
    let (client, device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        for case in CASES {
            let ops = operands(&client, &device, dtype, case);
            for with_bias in [false, true] {
                for with_residual in [false, true] {
                    let expected = composed(&client, &ops, case, act, with_bias, with_residual);
                    for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
                        let got = fused(&client, &ops, case, layout, act, with_bias, with_residual);
                        assert_tensor_allclose(
                            &got,
                            &expected,
                            dtype,
                            &format!(
                                "conv2d_fused {} {layout:?} {act:?} bias={with_bias} \
                                 residual={with_residual} vs composition [{dtype:?}]",
                                case.name
                            ),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_conv2d_fused_none_matches_composition() {
    cpu_matches_composition(Act::None);
}

#[test]
fn test_conv2d_fused_relu_matches_composition() {
    cpu_matches_composition(Act::Relu);
}

#[test]
fn test_conv2d_fused_leaky_relu_matches_composition() {
    cpu_matches_composition(Act::Leaky);
}

#[test]
fn test_conv2d_fused_prelu_per_channel_matches_composition() {
    cpu_matches_composition(Act::PreluPerChannel);
}

#[test]
fn test_conv2d_fused_prelu_shared_matches_composition() {
    cpu_matches_composition(Act::PreluShared);
}

/// Residual scale 1.0 takes the `add(residual, y)` branch with no scalar
/// multiply. The hand composition adds the residual to the activated conv.
#[test]
fn test_conv2d_fused_unit_residual_scale_matches_composition() {
    let (client, device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        for case in CASES {
            let ops = operands(&client, &device, dtype, case);
            for act in ACTS {
                for with_bias in [false, true] {
                    let y = composed(&client, &ops, case, act, with_bias, false);
                    let expected = client.add(&ops.residual, &y).unwrap();
                    for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
                        let got =
                            fused_scaled(&client, &ops, case, layout, act, with_bias, Some(1.0));
                        assert_tensor_allclose(
                            &got,
                            &expected,
                            dtype,
                            &format!(
                                "conv2d_fused {} {layout:?} {act:?} bias={with_bias} \
                                 residual scale=1.0 vs composition [{dtype:?}]",
                                case.name
                            ),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_conv2d_fused_nhwc_equals_nchw_permuted() {
    let (client, device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        for case in CASES {
            let ops = operands(&client, &device, dtype, case);
            for act in ACTS {
                let nchw = fused(&client, &ops, case, ConvLayout::Nchw, act, true, true);
                let nhwc = fused(&client, &ops, case, ConvLayout::Nhwc, act, true, true);
                assert_tensor_allclose(
                    &nhwc,
                    &nchw,
                    dtype,
                    &format!(
                        "conv2d_fused {} {act:?} NHWC vs NCHW [{dtype:?}]",
                        case.name
                    ),
                );
            }
        }
    }
}
