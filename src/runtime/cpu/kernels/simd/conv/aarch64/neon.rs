//! NEON convolution kernels for ARM64
//!
//! Provides vectorized implementations of conv1d, conv2d, and depthwise_conv2d
//! using 128-bit NEON registers.
//!
//! # SIMD Strategy
//!
//! Vectorizes over input channels using FMA instructions:
//! - f32: Process 4 input channels per iteration
//! - f64: Process 2 input channels per iteration
//!
//! For depthwise convolution (1 channel per group), vectorizes over output width.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::ops::conv_common::{Conv1dParams, Conv2dParams};

// ============================================================================
// Conv1d
// ============================================================================

/// NEON conv1d for f32
///
/// Vectorizes over input channels dimension.
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the operation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn conv1d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv1dParams,
) {
    let lanes = 4;
    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let chunks = c_in_per_group / lanes;

    for b in 0..params.batch_size {
        for g in 0..params.groups {
            let c_in_start = g * c_in_per_group;
            let c_out_start = g * c_out_per_group;

            for oc in 0..c_out_per_group {
                let out_c = c_out_start + oc;

                // Initialize with bias if present
                let bias_val = if let Some(b) = bias {
                    *b.add(out_c)
                } else {
                    0.0
                };

                for ol in 0..params.output_length {
                    let mut acc = vdupq_n_f32(bias_val);
                    let mut scalar_acc = 0.0f32;

                    for k in 0..params.kernel_size {
                        let il = ol * params.stride + k * params.dilation;

                        // Vectorized over input channels
                        for chunk in 0..chunks {
                            let ic_base = chunk * lanes;

                            // Load 4 input values
                            let in_idx = b * params.c_in * params.input_length
                                + (c_in_start + ic_base) * params.input_length
                                + il;

                            // Input is [batch, channels, length], need to gather
                            let in_arr = [
                                *input.add(in_idx),
                                *input.add(in_idx + params.input_length),
                                *input.add(in_idx + 2 * params.input_length),
                                *input.add(in_idx + 3 * params.input_length),
                            ];
                            let v_in = vld1q_f32(in_arr.as_ptr());

                            // Load 4 weight values
                            let w_idx = oc * c_in_per_group * params.kernel_size
                                + ic_base * params.kernel_size
                                + k;
                            let w_arr = [
                                *weight.add(w_idx),
                                *weight.add(w_idx + params.kernel_size),
                                *weight.add(w_idx + 2 * params.kernel_size),
                                *weight.add(w_idx + 3 * params.kernel_size),
                            ];
                            let v_w = vld1q_f32(w_arr.as_ptr());

                            // FMA
                            acc = vfmaq_f32(acc, v_in, v_w);
                        }

                        // Scalar tail for remaining channels
                        for ic in (chunks * lanes)..c_in_per_group {
                            let in_idx = b * params.c_in * params.input_length
                                + (c_in_start + ic) * params.input_length
                                + il;
                            let w_idx = oc * c_in_per_group * params.kernel_size
                                + ic * params.kernel_size
                                + k;
                            scalar_acc += *input.add(in_idx) * *weight.add(w_idx);
                        }
                    }

                    // Horizontal sum and store
                    let sum = vaddvq_f32(acc) + scalar_acc;
                    let out_idx =
                        b * params.c_out * params.output_length + out_c * params.output_length + ol;
                    *output.add(out_idx) = sum;
                }
            }
        }
    }
}

/// NEON conv1d for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn conv1d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv1dParams,
) {
    let lanes = 2;
    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let chunks = c_in_per_group / lanes;

    for b in 0..params.batch_size {
        for g in 0..params.groups {
            let c_in_start = g * c_in_per_group;
            let c_out_start = g * c_out_per_group;

            for oc in 0..c_out_per_group {
                let out_c = c_out_start + oc;

                let bias_val = if let Some(b) = bias {
                    *b.add(out_c)
                } else {
                    0.0
                };

                for ol in 0..params.output_length {
                    let mut acc = vdupq_n_f64(bias_val);
                    let mut scalar_acc = 0.0f64;

                    for k in 0..params.kernel_size {
                        let il = ol * params.stride + k * params.dilation;

                        for chunk in 0..chunks {
                            let ic_base = chunk * lanes;

                            let in_idx = b * params.c_in * params.input_length
                                + (c_in_start + ic_base) * params.input_length
                                + il;

                            let in_arr =
                                [*input.add(in_idx), *input.add(in_idx + params.input_length)];
                            let v_in = vld1q_f64(in_arr.as_ptr());

                            let w_idx = oc * c_in_per_group * params.kernel_size
                                + ic_base * params.kernel_size
                                + k;
                            let w_arr =
                                [*weight.add(w_idx), *weight.add(w_idx + params.kernel_size)];
                            let v_w = vld1q_f64(w_arr.as_ptr());

                            acc = vfmaq_f64(acc, v_in, v_w);
                        }

                        for ic in (chunks * lanes)..c_in_per_group {
                            let in_idx = b * params.c_in * params.input_length
                                + (c_in_start + ic) * params.input_length
                                + il;
                            let w_idx = oc * c_in_per_group * params.kernel_size
                                + ic * params.kernel_size
                                + k;
                            scalar_acc += *input.add(in_idx) * *weight.add(w_idx);
                        }
                    }

                    let sum = vaddvq_f64(acc) + scalar_acc;
                    let out_idx =
                        b * params.c_out * params.output_length + out_c * params.output_length + ol;
                    *output.add(out_idx) = sum;
                }
            }
        }
    }
}

// ============================================================================
// Conv2d
// ============================================================================

/// NEON conv2d for f32
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the operation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn conv2d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv2dParams,
) {
    let lanes = 4;
    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let chunks = c_in_per_group / lanes;

    let in_spatial = params.input_h * params.input_w;
    let out_spatial = params.output_h * params.output_w;
    let kernel_spatial = params.kernel_h * params.kernel_w;

    for b in 0..params.batch_size {
        for g in 0..params.groups {
            let c_in_start = g * c_in_per_group;
            let c_out_start = g * c_out_per_group;

            for oc in 0..c_out_per_group {
                let out_c = c_out_start + oc;

                let bias_val = if let Some(bias_ptr) = bias {
                    *bias_ptr.add(out_c)
                } else {
                    0.0
                };

                for oh in 0..params.output_h {
                    for ow in 0..params.output_w {
                        let mut acc = vdupq_n_f32(bias_val);
                        let mut scalar_acc = 0.0f32;

                        for kh in 0..params.kernel_h {
                            for kw in 0..params.kernel_w {
                                let ih = oh * params.stride_h + kh * params.dilation_h;
                                let iw = ow * params.stride_w + kw * params.dilation_w;

                                // Vectorize over input channels
                                for chunk in 0..chunks {
                                    let ic_base = chunk * lanes;

                                    // Gather 4 input values from different channels
                                    let mut in_arr = [0.0f32; 4];
                                    for lane in 0..lanes {
                                        let ic = ic_base + lane;
                                        let in_idx = b * params.c_in * in_spatial
                                            + (c_in_start + ic) * in_spatial
                                            + ih * params.input_w
                                            + iw;
                                        in_arr[lane] = *input.add(in_idx);
                                    }
                                    let v_in = vld1q_f32(in_arr.as_ptr());

                                    // Gather 4 weight values
                                    let mut w_arr = [0.0f32; 4];
                                    for lane in 0..lanes {
                                        let ic = ic_base + lane;
                                        let w_idx = oc * c_in_per_group * kernel_spatial
                                            + ic * kernel_spatial
                                            + kh * params.kernel_w
                                            + kw;
                                        w_arr[lane] = *weight.add(w_idx);
                                    }
                                    let v_w = vld1q_f32(w_arr.as_ptr());

                                    acc = vfmaq_f32(acc, v_in, v_w);
                                }

                                // Scalar tail
                                for ic in (chunks * lanes)..c_in_per_group {
                                    let in_idx = b * params.c_in * in_spatial
                                        + (c_in_start + ic) * in_spatial
                                        + ih * params.input_w
                                        + iw;
                                    let w_idx = oc * c_in_per_group * kernel_spatial
                                        + ic * kernel_spatial
                                        + kh * params.kernel_w
                                        + kw;
                                    scalar_acc += *input.add(in_idx) * *weight.add(w_idx);
                                }
                            }
                        }

                        let sum = vaddvq_f32(acc) + scalar_acc;
                        let out_idx = b * params.c_out * out_spatial
                            + out_c * out_spatial
                            + oh * params.output_w
                            + ow;
                        *output.add(out_idx) = sum;
                    }
                }
            }
        }
    }
}

/// NEON conv2d for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn conv2d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv2dParams,
) {
    let lanes = 2;
    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let chunks = c_in_per_group / lanes;

    let in_spatial = params.input_h * params.input_w;
    let out_spatial = params.output_h * params.output_w;
    let kernel_spatial = params.kernel_h * params.kernel_w;

    for b in 0..params.batch_size {
        for g in 0..params.groups {
            let c_in_start = g * c_in_per_group;
            let c_out_start = g * c_out_per_group;

            for oc in 0..c_out_per_group {
                let out_c = c_out_start + oc;

                let bias_val = if let Some(bias_ptr) = bias {
                    *bias_ptr.add(out_c)
                } else {
                    0.0
                };

                for oh in 0..params.output_h {
                    for ow in 0..params.output_w {
                        let mut acc = vdupq_n_f64(bias_val);
                        let mut scalar_acc = 0.0f64;

                        for kh in 0..params.kernel_h {
                            for kw in 0..params.kernel_w {
                                let ih = oh * params.stride_h + kh * params.dilation_h;
                                let iw = ow * params.stride_w + kw * params.dilation_w;

                                for chunk in 0..chunks {
                                    let ic_base = chunk * lanes;

                                    let mut in_arr = [0.0f64; 2];
                                    for lane in 0..lanes {
                                        let ic = ic_base + lane;
                                        let in_idx = b * params.c_in * in_spatial
                                            + (c_in_start + ic) * in_spatial
                                            + ih * params.input_w
                                            + iw;
                                        in_arr[lane] = *input.add(in_idx);
                                    }
                                    let v_in = vld1q_f64(in_arr.as_ptr());

                                    let mut w_arr = [0.0f64; 2];
                                    for lane in 0..lanes {
                                        let ic = ic_base + lane;
                                        let w_idx = oc * c_in_per_group * kernel_spatial
                                            + ic * kernel_spatial
                                            + kh * params.kernel_w
                                            + kw;
                                        w_arr[lane] = *weight.add(w_idx);
                                    }
                                    let v_w = vld1q_f64(w_arr.as_ptr());

                                    acc = vfmaq_f64(acc, v_in, v_w);
                                }

                                for ic in (chunks * lanes)..c_in_per_group {
                                    let in_idx = b * params.c_in * in_spatial
                                        + (c_in_start + ic) * in_spatial
                                        + ih * params.input_w
                                        + iw;
                                    let w_idx = oc * c_in_per_group * kernel_spatial
                                        + ic * kernel_spatial
                                        + kh * params.kernel_w
                                        + kw;
                                    scalar_acc += *input.add(in_idx) * *weight.add(w_idx);
                                }
                            }
                        }

                        let sum = vaddvq_f64(acc) + scalar_acc;
                        let out_idx = b * params.c_out * out_spatial
                            + out_c * out_spatial
                            + oh * params.output_w
                            + ow;
                        *output.add(out_idx) = sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Depthwise Conv2d
// ============================================================================

/// NEON depthwise conv2d for f32
///
/// For depthwise convolution, we vectorize over output width instead of channels.
///
/// # Safety
/// - All pointers must be valid and properly aligned
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn depthwise_conv2d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv2dParams,
) {
    let lanes = 4;
    let out_w_chunks = params.output_w / lanes;

    let in_spatial = params.input_h * params.input_w;
    let out_spatial = params.output_h * params.output_w;
    let kernel_spatial = params.kernel_h * params.kernel_w;

    for b in 0..params.batch_size {
        for c in 0..params.c_out {
            let bias_val = if let Some(bias_ptr) = bias {
                *bias_ptr.add(c)
            } else {
                0.0
            };
            let v_bias = vdupq_n_f32(bias_val);

            for oh in 0..params.output_h {
                // Vectorized output width
                for ow_chunk in 0..out_w_chunks {
                    let ow_base = ow_chunk * lanes;
                    let mut acc = v_bias;

                    for kh in 0..params.kernel_h {
                        for kw in 0..params.kernel_w {
                            let w_val = *weight.add(c * kernel_spatial + kh * params.kernel_w + kw);
                            let v_w = vdupq_n_f32(w_val);

                            let ih = oh * params.stride_h + kh * params.dilation_h;

                            // Gather 4 input values at different output width positions
                            let mut in_arr = [0.0f32; 4];
                            for lane in 0..lanes {
                                let ow = ow_base + lane;
                                let iw = ow * params.stride_w + kw * params.dilation_w;
                                let in_idx = b * params.c_in * in_spatial
                                    + c * in_spatial
                                    + ih * params.input_w
                                    + iw;
                                in_arr[lane] = *input.add(in_idx);
                            }
                            let v_in = vld1q_f32(in_arr.as_ptr());

                            acc = vfmaq_f32(acc, v_in, v_w);
                        }
                    }

                    // Store 4 output values
                    let out_idx = b * params.c_out * out_spatial
                        + c * out_spatial
                        + oh * params.output_w
                        + ow_base;
                    vst1q_f32(output.add(out_idx), acc);
                }

                // Scalar tail for remaining output width
                for ow in (out_w_chunks * lanes)..params.output_w {
                    let mut sum = bias_val;

                    for kh in 0..params.kernel_h {
                        for kw in 0..params.kernel_w {
                            let ih = oh * params.stride_h + kh * params.dilation_h;
                            let iw = ow * params.stride_w + kw * params.dilation_w;

                            let in_idx = b * params.c_in * in_spatial
                                + c * in_spatial
                                + ih * params.input_w
                                + iw;
                            let w_idx = c * kernel_spatial + kh * params.kernel_w + kw;

                            sum += *input.add(in_idx) * *weight.add(w_idx);
                        }
                    }

                    let out_idx = b * params.c_out * out_spatial
                        + c * out_spatial
                        + oh * params.output_w
                        + ow;
                    *output.add(out_idx) = sum;
                }
            }
        }
    }
}

/// NEON depthwise conv2d for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn depthwise_conv2d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv2dParams,
) {
    let lanes = 2;
    let out_w_chunks = params.output_w / lanes;

    let in_spatial = params.input_h * params.input_w;
    let out_spatial = params.output_h * params.output_w;
    let kernel_spatial = params.kernel_h * params.kernel_w;

    for b in 0..params.batch_size {
        for c in 0..params.c_out {
            let bias_val = if let Some(bias_ptr) = bias {
                *bias_ptr.add(c)
            } else {
                0.0
            };
            let v_bias = vdupq_n_f64(bias_val);

            for oh in 0..params.output_h {
                for ow_chunk in 0..out_w_chunks {
                    let ow_base = ow_chunk * lanes;
                    let mut acc = v_bias;

                    for kh in 0..params.kernel_h {
                        for kw in 0..params.kernel_w {
                            let w_val = *weight.add(c * kernel_spatial + kh * params.kernel_w + kw);
                            let v_w = vdupq_n_f64(w_val);

                            let ih = oh * params.stride_h + kh * params.dilation_h;

                            let mut in_arr = [0.0f64; 2];
                            for lane in 0..lanes {
                                let ow = ow_base + lane;
                                let iw = ow * params.stride_w + kw * params.dilation_w;
                                let in_idx = b * params.c_in * in_spatial
                                    + c * in_spatial
                                    + ih * params.input_w
                                    + iw;
                                in_arr[lane] = *input.add(in_idx);
                            }
                            let v_in = vld1q_f64(in_arr.as_ptr());

                            acc = vfmaq_f64(acc, v_in, v_w);
                        }
                    }

                    let out_idx = b * params.c_out * out_spatial
                        + c * out_spatial
                        + oh * params.output_w
                        + ow_base;
                    vst1q_f64(output.add(out_idx), acc);
                }

                for ow in (out_w_chunks * lanes)..params.output_w {
                    let mut sum = bias_val;

                    for kh in 0..params.kernel_h {
                        for kw in 0..params.kernel_w {
                            let ih = oh * params.stride_h + kh * params.dilation_h;
                            let iw = ow * params.stride_w + kw * params.dilation_w;

                            let in_idx = b * params.c_in * in_spatial
                                + c * in_spatial
                                + ih * params.input_w
                                + iw;
                            let w_idx = c * kernel_spatial + kh * params.kernel_w + kw;

                            sum += *input.add(in_idx) * *weight.add(w_idx);
                        }
                    }

                    let out_idx = b * params.c_out * out_spatial
                        + c * out_spatial
                        + oh * params.output_w
                        + ow;
                    *output.add(out_idx) = sum;
                }
            }
        }
    }
}
