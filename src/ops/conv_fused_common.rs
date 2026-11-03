//! Shared validation for fused 2D convolution (`FusedConvOps`).
//!
//! NHWC shapes map to NCHW order, then [`validate_conv2d`] checks the
//! geometry. Every backend calls these functions, so every backend rejects
//! the same inputs with the same error.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::conv_common::{Conv2dParams, validate_4d_tensor, validate_conv2d};
use crate::ops::traits::{Conv2dConfig, ConvActivation, ConvLayout, ConvResidual};
use crate::runtime::Runtime;
use crate::runtime::common::view_span::{check_out_disjoint_from, has_broadcast_dim};
use crate::tensor::Tensor;

const OP: &str = "conv2d_fused";

/// Permutation from NHWC to NCHW, and from OHWI to OIHW.
pub const NHWC_TO_NCHW: [usize; 4] = [0, 3, 1, 2];

/// Permutation from NCHW to NHWC.
pub const NCHW_TO_NHWC: [usize; 4] = [0, 2, 3, 1];

/// Validated geometry of one fused convolution.
#[derive(Debug, Clone, Copy)]
pub struct FusedConv2dParams {
    /// Geometry in NCHW terms, whatever the layout.
    pub conv: Conv2dParams,
    /// Layout of input, weight, output and residual.
    pub layout: ConvLayout,
    /// Output shape in `layout`.
    pub output_shape: [usize; 4],
}

/// Reorder a 4D shape by `perm`.
pub fn permute_shape(shape: [usize; 4], perm: [usize; 4]) -> [usize; 4] {
    perm.map(|axis| shape[axis])
}

/// Read a shape as `[usize; 4]`, rejecting any other rank.
fn shape_4d(shape: &[usize], arg: &'static str) -> Result<[usize; 4]> {
    validate_4d_tensor(shape, arg, OP)?;
    let mut dims = [0usize; 4];
    for (dim, &size) in dims.iter_mut().zip(shape) {
        *dim = size;
    }
    Ok(dims)
}

/// Shape in NCHW (input) or OIHW (weight) order.
fn to_channels_first(layout: ConvLayout, shape: [usize; 4]) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => shape,
        ConvLayout::Nhwc => permute_shape(shape, NHWC_TO_NCHW),
    }
}

/// Check the weight's input-channel count, naming the dimension of its layout.
///
/// `validate_conv2d` names `weight.shape[1]`, which is the OIHW position. An
/// OHWI weight holds input channels at `weight.shape[3]`. Groups that do not
/// divide `c_in` are left to `validate_conv2d`.
fn check_weight_channels(config: &Conv2dConfig, c_in: usize, weight_c_in: usize) -> Result<()> {
    let groups = config.groups;
    if groups == 0 || !c_in.is_multiple_of(groups) || weight_c_in == c_in / groups {
        return Ok(());
    }
    let dim = match config.layout {
        ConvLayout::Nchw => 1,
        ConvLayout::Nhwc => 3,
    };
    Err(Error::InvalidArgument {
        arg: "weight",
        reason: format!(
            "{OP}: weight.shape[{dim}] must be C_in/groups = {c_in}/{groups} = {}, got {weight_c_in}",
            c_in / groups
        ),
    })
}

/// Check every read operand and compute the output geometry.
///
/// # Errors
///
/// Returns the errors listed on `FusedConvOps::conv2d_fused`.
pub fn validate_conv2d_fused<R: Runtime<DType = DType>>(
    input: &Tensor<R>,
    weight: &Tensor<R>,
    bias: Option<&Tensor<R>>,
    config: &Conv2dConfig,
    activation: ConvActivation<'_, R>,
    residual: Option<ConvResidual<'_, R>>,
) -> Result<FusedConv2dParams> {
    let input_nchw = to_channels_first(config.layout, shape_4d(input.shape(), "input")?);
    let weight_oihw = to_channels_first(config.layout, shape_4d(weight.shape(), "weight")?);
    check_weight_channels(config, input_nchw[1], weight_oihw[1])?;
    let conv = validate_conv2d(
        &input_nchw,
        &weight_oihw,
        bias.map(|b| b.shape()),
        config.stride,
        config.padding,
        config.dilation,
        config.groups,
        input.dtype(),
        weight.dtype(),
        bias.map(|b| b.dtype()),
    )?;
    let output_nchw = [conv.batch, conv.c_out, conv.output_h, conv.output_w];
    let output_shape = match config.layout {
        ConvLayout::Nchw => output_nchw,
        ConvLayout::Nhwc => permute_shape(output_nchw, NCHW_TO_NHWC),
    };
    validate_activation(activation, input.dtype(), conv.c_out)?;
    if let Some(residual) = residual {
        validate_residual(residual, input.dtype(), &output_shape)?;
    }
    Ok(FusedConv2dParams {
        conv,
        layout: config.layout,
        output_shape,
    })
}

/// Check a `Prelu` weight or `LeakyRelu` slope.
fn validate_activation<R: Runtime<DType = DType>>(
    activation: ConvActivation<'_, R>,
    dtype: DType,
    c_out: usize,
) -> Result<()> {
    match activation {
        ConvActivation::None | ConvActivation::Relu => Ok(()),
        ConvActivation::LeakyRelu(slope) if slope.is_finite() => Ok(()),
        ConvActivation::LeakyRelu(slope) => Err(Error::InvalidArgument {
            arg: "activation",
            reason: format!("{OP}: LeakyRelu slope must be finite, got {slope}"),
        }),
        ConvActivation::Prelu(weight) => {
            if weight.dtype() != dtype {
                return Err(Error::DTypeMismatch {
                    lhs: dtype,
                    rhs: weight.dtype(),
                });
            }
            if weight.shape() != [c_out] && weight.shape() != [1] {
                return Err(Error::InvalidArgument {
                    arg: "activation",
                    reason: format!(
                        "{OP}: Prelu weight must be [{c_out}] (C_out) or [1], got {:?}",
                        weight.shape()
                    ),
                });
            }
            Ok(())
        }
    }
}

/// Check the residual's dtype, shape and scale.
fn validate_residual<R: Runtime<DType = DType>>(
    residual: ConvResidual<'_, R>,
    dtype: DType,
    output_shape: &[usize; 4],
) -> Result<()> {
    if residual.tensor.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: residual.tensor.dtype(),
        });
    }
    if residual.tensor.shape() != output_shape {
        return Err(Error::shape_mismatch(output_shape, residual.tensor.shape()));
    }
    if !residual.scale.is_finite() {
        return Err(Error::InvalidArgument {
            arg: "residual",
            reason: format!(
                "{OP}: residual scale must be finite, got {}",
                residual.scale
            ),
        });
    }
    Ok(())
}

/// Check the destination of `conv2d_fused_into`.
///
/// `out` must have the output shape and the input's dtype, must not repeat an
/// element, and must not overlap any read operand. See
/// [`check_out_disjoint_from`] for the overlap rule.
///
/// # Errors
///
/// Returns `Error::ShapeMismatch`, `Error::DTypeMismatch`, or
/// `Error::InvalidArgument` for argument `out`.
pub fn validate_conv2d_fused_out<R: Runtime<DType = DType>>(
    out: &Tensor<R>,
    params: &FusedConv2dParams,
    input: &Tensor<R>,
    weight: &Tensor<R>,
    bias: Option<&Tensor<R>>,
    activation: ConvActivation<'_, R>,
    residual: Option<ConvResidual<'_, R>>,
) -> Result<()> {
    if out.dtype() != input.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: input.dtype(),
            rhs: out.dtype(),
        });
    }
    if out.shape() != params.output_shape {
        return Err(Error::shape_mismatch(&params.output_shape, out.shape()));
    }
    if has_broadcast_dim(out.shape(), out.strides()) {
        return Err(Error::InvalidArgument {
            arg: "out",
            reason: format!("{OP}: out repeats elements (stride 0 on a dimension > 1)"),
        });
    }
    check_out_disjoint_from(out, input, "input", OP)?;
    check_out_disjoint_from(out, weight, "weight", OP)?;
    if let Some(bias) = bias {
        check_out_disjoint_from(out, bias, "bias", OP)?;
    }
    if let ConvActivation::Prelu(prelu_weight) = activation {
        check_out_disjoint_from(out, prelu_weight, "prelu weight", OP)?;
    }
    if let Some(residual) = residual {
        check_out_disjoint_from(out, residual.tensor, "residual", OP)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::PaddingMode;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn tensor(shape: &[usize], dtype: DType) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, dtype, &CpuDevice::new()).unwrap()
    }

    fn f32(shape: &[usize]) -> Tensor<CpuRuntime> {
        tensor(shape, DType::F32)
    }

    fn nhwc_same() -> Conv2dConfig {
        Conv2dConfig::new(ConvLayout::Nhwc).with_padding(PaddingMode::Same)
    }

    fn validate(
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, CpuRuntime>,
        residual: Option<ConvResidual<'_, CpuRuntime>>,
    ) -> Result<FusedConv2dParams> {
        validate_conv2d_fused(input, weight, bias, config, activation, residual)
    }

    fn assert_invalid(result: Result<impl std::fmt::Debug>, expected_arg: &str) {
        match result {
            Err(Error::InvalidArgument { arg, .. }) => assert_eq!(arg, expected_arg),
            other => panic!("expected InvalidArgument for {expected_arg}, got {other:?}"),
        }
    }

    #[test]
    fn nhwc_shapes_map_to_nchw() {
        let input = f32(&[2, 5, 6, 4]);
        let weight = f32(&[8, 3, 3, 4]);
        let p = validate(
            &input,
            &weight,
            None,
            &nhwc_same(),
            ConvActivation::None,
            None,
        )
        .unwrap();
        assert_eq!(p.conv.batch, 2);
        assert_eq!(p.conv.c_in, 4);
        assert_eq!(p.conv.height, 5);
        assert_eq!(p.conv.width, 6);
        assert_eq!(p.conv.c_out, 8);
        assert_eq!((p.conv.kernel_h, p.conv.kernel_w), (3, 3));
        assert_eq!(p.output_shape, [2, 5, 6, 8]);
        assert_eq!(p.layout, ConvLayout::Nhwc);
    }

    #[test]
    fn nchw_output_shape_with_stride() {
        let input = f32(&[1, 4, 9, 8]);
        let weight = f32(&[6, 4, 3, 3]);
        let config = Conv2dConfig::new(ConvLayout::Nchw).with_stride((2, 2));
        let p = validate(&input, &weight, None, &config, ConvActivation::Relu, None).unwrap();
        assert_eq!(p.output_shape, [1, 6, 4, 3]);
    }

    #[test]
    fn nhwc_grouped_dilated_output_shape() {
        let input = f32(&[1, 7, 7, 4]);
        let weight = f32(&[6, 3, 3, 2]);
        let config = Conv2dConfig::new(ConvLayout::Nhwc)
            .with_dilation((2, 2))
            .with_groups(2);
        let p = validate(&input, &weight, None, &config, ConvActivation::None, None).unwrap();
        assert_eq!(p.output_shape, [1, 3, 3, 6]);
    }

    #[test]
    fn permutations_round_trip() {
        let s = [2, 3, 4, 5];
        assert_eq!(
            permute_shape(permute_shape(s, NHWC_TO_NCHW), NCHW_TO_NHWC),
            s
        );
        assert_eq!(permute_shape(s, NHWC_TO_NCHW), [2, 5, 3, 4]);
    }

    #[test]
    fn rejects_non_4d_input() {
        let r = validate(
            &f32(&[5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            None,
            &nhwc_same(),
            ConvActivation::None,
            None,
        );
        assert_invalid(r, "input");
    }

    #[test]
    fn rejects_non_4d_weight() {
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 9, 4]),
            None,
            &nhwc_same(),
            ConvActivation::None,
            None,
        );
        assert_invalid(r, "weight");
    }

    #[test]
    fn rejects_nhwc_weight_channel_mismatch() {
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 3]),
            None,
            &nhwc_same(),
            ConvActivation::None,
            None,
        );
        match r {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "weight");
                assert!(reason.contains("weight.shape[3]"), "{reason}");
            }
            other => panic!("expected InvalidArgument for weight, got {other:?}"),
        }
    }

    #[test]
    fn nchw_weight_channel_mismatch_names_dim_1() {
        let r = validate(
            &f32(&[1, 4, 5, 6]),
            &f32(&[8, 3, 3, 3]),
            None,
            &Conv2dConfig::new(ConvLayout::Nchw),
            ConvActivation::None,
            None,
        );
        match r {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "weight");
                assert!(reason.contains("weight.shape[1]"), "{reason}");
            }
            other => panic!("expected InvalidArgument for weight, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_groups() {
        let config = nhwc_same().with_groups(0);
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            None,
            &config,
            ConvActivation::None,
            None,
        );
        assert_invalid(r, "groups");
    }

    #[test]
    fn rejects_bias_of_wrong_length() {
        let bias = f32(&[7]);
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            Some(&bias),
            &nhwc_same(),
            ConvActivation::None,
            None,
        );
        assert_invalid(r, "bias");
    }

    #[test]
    fn rejects_integer_dtype() {
        let r = validate(
            &tensor(&[1, 5, 6, 4], DType::I32),
            &tensor(&[8, 3, 3, 4], DType::I32),
            None,
            &nhwc_same(),
            ConvActivation::None,
            None,
        );
        assert!(matches!(r, Err(Error::UnsupportedDType { .. })), "{r:?}");
    }

    #[test]
    fn accepts_prelu_of_c_out_or_one() {
        let input = f32(&[1, 5, 6, 4]);
        let weight = f32(&[8, 3, 3, 4]);
        for shape in [[8usize], [1]] {
            let slope = f32(&shape);
            let act = ConvActivation::Prelu(&slope);
            assert!(validate(&input, &weight, None, &nhwc_same(), act, None).is_ok());
        }
    }

    #[test]
    fn rejects_prelu_of_wrong_shape() {
        let slope = f32(&[4]);
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            None,
            &nhwc_same(),
            ConvActivation::Prelu(&slope),
            None,
        );
        assert_invalid(r, "activation");
    }

    #[test]
    fn rejects_prelu_of_wrong_dtype() {
        let slope = tensor(&[8], DType::F64);
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            None,
            &nhwc_same(),
            ConvActivation::Prelu(&slope),
            None,
        );
        assert!(matches!(r, Err(Error::DTypeMismatch { .. })), "{r:?}");
    }

    #[test]
    fn rejects_non_finite_leaky_slope() {
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            None,
            &nhwc_same(),
            ConvActivation::LeakyRelu(f64::NAN),
            None,
        );
        assert_invalid(r, "activation");
    }

    #[test]
    fn rejects_residual_of_wrong_shape() {
        let res = f32(&[1, 8, 5, 6]);
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            None,
            &nhwc_same(),
            ConvActivation::None,
            Some(ConvResidual {
                tensor: &res,
                scale: 0.2,
            }),
        );
        assert!(matches!(r, Err(Error::ShapeMismatch { .. })), "{r:?}");
    }

    #[test]
    fn rejects_residual_of_wrong_dtype() {
        let res = tensor(&[1, 5, 6, 8], DType::F64);
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            None,
            &nhwc_same(),
            ConvActivation::None,
            Some(ConvResidual {
                tensor: &res,
                scale: 0.2,
            }),
        );
        assert!(matches!(r, Err(Error::DTypeMismatch { .. })), "{r:?}");
    }

    #[test]
    fn rejects_non_finite_residual_scale() {
        let res = f32(&[1, 5, 6, 8]);
        let r = validate(
            &f32(&[1, 5, 6, 4]),
            &f32(&[8, 3, 3, 4]),
            None,
            &nhwc_same(),
            ConvActivation::None,
            Some(ConvResidual {
                tensor: &res,
                scale: f64::INFINITY,
            }),
        );
        assert_invalid(r, "residual");
    }

    /// Validate `out` for a `[1, 5, 6, 4]` NHWC input and `[2, 3, 3, 4]` weight.
    fn validate_out(
        out: &Tensor<CpuRuntime>,
        input: &Tensor<CpuRuntime>,
        residual: Option<&Tensor<CpuRuntime>>,
    ) -> Result<()> {
        let weight = f32(&[2, 3, 3, 4]);
        let residual = residual.map(|tensor| ConvResidual { tensor, scale: 0.2 });
        let params = validate(
            input,
            &weight,
            None,
            &nhwc_same(),
            ConvActivation::None,
            residual,
        )?;
        validate_conv2d_fused_out(
            out,
            &params,
            input,
            &weight,
            None,
            ConvActivation::None,
            residual,
        )
    }

    #[test]
    fn out_accepts_disjoint_slice_of_input_buffer() {
        let buf = f32(&[1, 5, 6, 6]);
        let input = buf.narrow(-1, 0, 4).unwrap();
        let out = buf.narrow(-1, 4, 2).unwrap();
        assert!(validate_out(&out, &input, None).is_ok());
    }

    #[test]
    fn out_rejects_wrong_shape() {
        let out = f32(&[1, 5, 6, 3]);
        let r = validate_out(&out, &f32(&[1, 5, 6, 4]), None);
        assert!(matches!(r, Err(Error::ShapeMismatch { .. })), "{r:?}");
    }

    #[test]
    fn out_rejects_wrong_dtype() {
        let out = tensor(&[1, 5, 6, 2], DType::F64);
        let r = validate_out(&out, &f32(&[1, 5, 6, 4]), None);
        assert!(matches!(r, Err(Error::DTypeMismatch { .. })), "{r:?}");
    }

    #[test]
    fn out_rejects_broadcast_view() {
        let out = f32(&[1, 1, 1, 2]).broadcast_to(&[1, 5, 6, 2]).unwrap();
        let r = validate_out(&out, &f32(&[1, 5, 6, 4]), None);
        assert_invalid(r, "out");
    }

    #[test]
    fn out_rejects_overlap_with_input() {
        let buf = f32(&[1, 5, 6, 6]);
        let input = buf.narrow(-1, 0, 4).unwrap();
        let out = buf.narrow(-1, 3, 2).unwrap();
        match validate_out(&out, &input, None) {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "out");
                assert!(reason.contains("input"), "{reason}");
            }
            other => panic!("expected overlap error, got {other:?}"),
        }
    }

    #[test]
    fn out_rejects_overlap_with_residual() {
        let buf = f32(&[1, 5, 6, 4]);
        let out = buf.narrow(-1, 0, 2).unwrap();
        let residual = buf.narrow(-1, 1, 2).unwrap();
        match validate_out(&out, &f32(&[1, 5, 6, 4]), Some(&residual)) {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "out");
                assert!(reason.contains("residual"), "{reason}");
            }
            other => panic!("expected overlap error, got {other:?}"),
        }
    }

    #[test]
    fn out_accepts_residual_disjoint_in_same_buffer() {
        let buf = f32(&[1, 5, 6, 4]);
        let out = buf.narrow(-1, 2, 2).unwrap();
        let residual = buf.narrow(-1, 0, 2).unwrap();
        assert!(validate_out(&out, &f32(&[1, 5, 6, 4]), Some(&residual)).is_ok());
    }
}
