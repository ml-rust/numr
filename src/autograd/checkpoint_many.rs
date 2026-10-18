//! Activation checkpointing for a segment with several outputs.
//!
//! [`checkpoint_many_with_client`] packs the outputs into one 1-D tensor. It
//! checkpoints the packed form with [`checkpoint_with_client`], then unpacks
//! it into one `Var` per output.
//!
//! # Example
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, backward, checkpoint_many_with_client, var_add, var_mul, var_sum};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! let x = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device)?, true);
//! let w = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device)?, true);
//!
//! // A main output and an auxiliary scalar from one segment.
//! let outs = checkpoint_many_with_client(
//!     |xs: &[Var<CpuRuntime>], c: &CpuClient| {
//!         let main = var_mul(&xs[0], &xs[1], c)?;
//!         let aux = var_sum(&var_mul(&xs[1], &xs[1], c)?, &[], false, c)?;
//!         Ok(vec![main, aux])
//!     },
//!     &[&x, &w],
//!     &client,
//! )?;
//!
//! let main = var_sum(&outs[0], &[], false, &client)?;
//! let loss = var_add(&main, &outs[1], &client)?;
//! let grads = backward(&loss, &client)?;
//! // grad_w = x + 2w = 7
//! # Ok::<(), numr::error::Error>(())
//! ```

use std::sync::{Arc, Mutex};

use super::checkpoint::checkpoint_with_client;
use super::checkpoint_many_record::{Record, Slot, describe, recorded, remember};
use crate::autograd::{Var, var_cast, var_cat, var_narrow, var_reshape};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};

/// Run `f` on `inputs` with activation checkpointing, for a segment with several outputs.
///
/// This behaves as [`checkpoint_with_client`], except that `f` returns a `Vec`.
/// The result holds one `Var` per output, in the same order. Each keeps its
/// shape and dtype. An output the loss reads carries its gradient into the
/// segment, even when no other output depends on it.
///
/// # Packing
///
/// The wrapper flattens every output and casts it to the widest float dtype
/// present. The order is F64, then F32, then F16 or BF16. F16 mixed with BF16
/// packs as F32. FP8E4M3 mixed with FP8E5M2 packs as F16. Every widening cast
/// is exact, so forward values and gradients keep their bits. One exception
/// exists: a negative-zero output gradient can reach the segment as positive zero.
///
/// The first forward records each output's shape and dtype. The backward
/// recompute must return the same shapes and dtypes.
///
/// # Errors
///
/// - `inputs` is empty.
/// - `f` returns no outputs.
/// - An output has a non-float dtype, such as an integer or Bool type.
/// - The recompute returns shapes or dtypes that differ from the first forward.
///   This error surfaces from `backward`.
pub fn checkpoint_many_with_client<R, C, F>(
    f: F,
    inputs: &[&Var<R>],
    client: &C,
) -> Result<Vec<Var<R>>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + 'static,
    R::Client: TensorOps<R>,
    F: Fn(&[Var<R>], &C) -> Result<Vec<Var<R>>> + Send + Sync + 'static,
{
    let record: Record = Arc::new(Mutex::new(None));
    let recorder = Arc::clone(&record);

    let packed = checkpoint_with_client(
        move |xs: &[Var<R>], c: &C| {
            let outputs = f(xs, c)?;
            let slots = describe(&outputs)?;
            remember(&recorder, &slots)?;
            pack(&outputs, packed_dtype(&slots), c)
        },
        inputs,
        client,
    )?;

    let slots = recorded(&record)?;
    unpack(&packed, &slots, client)
}

/// The narrowest float dtype that holds every output dtype exactly.
fn packed_dtype(slots: &[Slot]) -> DType {
    let has = |dtype: DType| slots.iter().any(|slot| slot.dtype == dtype);
    if has(DType::F64) {
        DType::F64
    } else if has(DType::F32) || (has(DType::F16) && has(DType::BF16)) {
        DType::F32
    } else if has(DType::F16) {
        DType::F16
    } else if has(DType::BF16) {
        DType::BF16
    } else if has(DType::FP8E4M3) && has(DType::FP8E5M2) {
        DType::F16
    } else {
        // All outputs share one FP8 dtype. `describe` refuses an empty list.
        slots.first().map_or(DType::F32, |slot| slot.dtype)
    }
}

/// Cast every output to `dtype`, flatten it, and join the results into one 1-D `Var`.
fn pack<R, C>(outputs: &[Var<R>], dtype: DType, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let flats = outputs
        .iter()
        .map(|output| flatten(output, dtype, client))
        .collect::<Result<Vec<_>>>()?;
    match <[Var<R>; 1]>::try_from(flats) {
        Ok([only]) => Ok(only),
        Err(flats) => {
            let parts: Vec<&Var<R>> = flats.iter().collect();
            var_cat(&parts, 0, client)
        }
    }
}

/// Cast `output` to `dtype` and reshape it to 1-D.
fn flatten<R, C>(output: &Var<R>, dtype: DType, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let wide = var_cast(output, dtype, client)?;
    // A reshape needs contiguous memory. A one-input cat copies the view into
    // fresh memory and passes the gradient back unchanged.
    let dense = if wide.tensor().is_contiguous() {
        wide
    } else {
        var_cat(&[&wide], 0, client)?
    };
    var_reshape(&dense, &[dense.numel()])
}

/// Split the packed `Var` back into one `Var` per slot, with its shape and dtype.
fn unpack<R, C>(packed: &Var<R>, slots: &[Slot], client: &C) -> Result<Vec<Var<R>>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let mut start = 0;
    slots
        .iter()
        .map(|slot| {
            let length = slot.numel();
            let piece = var_narrow(packed, 0, start, length)?;
            start += length;
            let shaped = var_reshape(&piece, &slot.shape)?;
            var_cast(&shaped, slot.dtype, client)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{backward, var_add, var_exp, var_mul, var_sum, var_transpose};
    use crate::dtype::Element;
    use crate::ops::TypeConversionOps;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Segment = fn(&[Var<CpuRuntime>], &CpuClient) -> Result<Vec<Var<CpuRuntime>>>;

    fn cpu() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    fn leaf<T: Element>(values: &[T], shape: &[usize], device: &CpuDevice) -> Var<CpuRuntime> {
        let tensor = Tensor::<CpuRuntime>::from_slice(values, shape, device).unwrap();
        Var::new(tensor, true)
    }

    /// Bits of `t` widened to F64. The widening is exact and injective.
    fn bits(t: &Tensor<CpuRuntime>, client: &CpuClient) -> Vec<u64> {
        let wide = client.cast(t, DType::F64).unwrap().contiguous().unwrap();
        wide.to_vec::<f64>().iter().map(|v| v.to_bits()).collect()
    }

    /// `sum_i sum(o_i * o_i)`, accumulated in F64.
    fn loss(outputs: &[Var<CpuRuntime>], c: &CpuClient) -> Var<CpuRuntime> {
        let mut total: Option<Var<CpuRuntime>> = None;
        for output in outputs {
            let square = var_mul(output, output, c).unwrap();
            let sum = var_sum(&square, &[], false, c).unwrap();
            let term = var_cast(&sum, DType::F64, c).unwrap();
            total = Some(match total {
                None => term,
                Some(t) => var_add(&t, &term, c).unwrap(),
            });
        }
        total.unwrap()
    }

    /// Run `segment` plain and checkpointed. Output and gradient bits must match.
    fn check_against_plain(segment: Segment, inputs: &[&Var<CpuRuntime>], client: &CpuClient) {
        let plain_inputs: Vec<Var<CpuRuntime>> = inputs.iter().map(|v| v.alias()).collect();
        let plain = segment(&plain_inputs, client).unwrap();
        let plain_grads = backward(&loss(&plain, client), client).unwrap();

        let outs = checkpoint_many_with_client(segment, inputs, client).unwrap();
        let grads = backward(&loss(&outs, client), client).unwrap();

        assert_eq!(outs.len(), plain.len());
        for (i, (got, want)) in outs.iter().zip(&plain).enumerate() {
            assert_eq!(
                got.tensor().dtype(),
                want.tensor().dtype(),
                "output {i} dtype"
            );
            assert_eq!(got.shape(), want.shape(), "output {i} shape");
            assert_eq!(
                bits(got.tensor(), client),
                bits(want.tensor(), client),
                "output {i} values"
            );
        }
        for (i, input) in inputs.iter().enumerate() {
            let got = grads.get(input.id()).expect("checkpointed gradient");
            let want = plain_grads.get(input.id()).expect("plain gradient");
            assert_eq!(got.dtype(), want.dtype(), "input {i} gradient dtype");
            assert_eq!(bits(got, client), bits(want, client), "input {i} gradient");
        }
    }

    fn segment_f32(xs: &[Var<CpuRuntime>], c: &CpuClient) -> Result<Vec<Var<CpuRuntime>>> {
        let xw = var_mul(&xs[0], &xs[1], c)?;
        let rows = var_sum(&var_exp(&xs[0], c)?, &[1], false, c)?;
        // A transposed view is not contiguous, so packing must copy it.
        let turned = var_transpose(&xw)?;
        Ok(vec![xw, rows, turned])
    }

    #[test]
    fn test_checkpoint_many_f32_matches_plain() {
        let (device, client) = cpu();
        // Grid values keep every product and sum exact in any order.
        let x = leaf(&[0.25f32, -0.5, 0.75, 1.0, -1.25, 0.5], &[2, 3], &device);
        let w = leaf(&[1.5f32, 0.25, -0.75, 0.5, 1.0, -0.25], &[2, 3], &device);
        check_against_plain(segment_f32, &[&x, &w], &client);
    }

    #[cfg(feature = "f16")]
    fn segment_bf16_f32(xs: &[Var<CpuRuntime>], c: &CpuClient) -> Result<Vec<Var<CpuRuntime>>> {
        let square = var_cast(&var_mul(&xs[0], &xs[0], c)?, DType::BF16, c)?;
        let scaled = crate::autograd::var_mul_scalar(&xs[0], 3.0, c)?;
        Ok(vec![square, scaled])
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_checkpoint_many_bf16_and_f32_match_plain() {
        let (device, client) = cpu();
        // Values that BF16 rounds, so a lossy pack would change bits.
        let x = leaf(&[0.3f32, -1.7, 2.9, 0.011], &[4], &device);
        check_against_plain(segment_bf16_f32, &[&x], &client);
    }

    fn segment_f64(xs: &[Var<CpuRuntime>], c: &CpuClient) -> Result<Vec<Var<CpuRuntime>>> {
        let xw = var_mul(&xs[0], &xs[1], c)?;
        let column = var_reshape(&var_exp(&xs[0], c)?, &[3, 1])?;
        let scalar = var_sum(&var_mul(&xs[1], &xs[1], c)?, &[], false, c)?;
        Ok(vec![xw, column, scalar])
    }

    #[test]
    fn test_checkpoint_many_f64_matches_plain() {
        let (device, client) = cpu();
        let x = leaf(&[0.3f64, -1.7, 2.9], &[3], &device);
        let w = leaf(&[1.1f64, 0.07, -0.6], &[3], &device);
        check_against_plain(segment_f64, &[&x, &w], &client);
    }

    fn aux_segment(
        xs: &[Var<CpuRuntime>],
        p: &Var<CpuRuntime>,
        c: &CpuClient,
    ) -> Result<Vec<Var<CpuRuntime>>> {
        let main = var_mul(&xs[0], &xs[1], c)?;
        let aux = var_sum(&var_mul(p, p, c)?, &[], false, c)?;
        Ok(vec![main, aux])
    }

    #[test]
    fn test_checkpoint_many_aux_output_trains_its_parameter() {
        // `p` reaches only the aux scalar, and the segment captures it unlisted.
        let (device, client) = cpu();
        let x = leaf(&[0.3f32, -1.7, 2.9], &[3], &device);
        let w = leaf(&[1.1f32, 0.07, -0.6], &[3], &device);
        let p = leaf(&[0.45f32, -2.25], &[2], &device);
        let task_loss = |outs: &[Var<CpuRuntime>]| {
            let main = var_sum(&outs[0], &[], false, &client).unwrap();
            var_add(&main, &outs[1], &client).unwrap()
        };

        let plain = aux_segment(&[x.alias(), w.alias()], &p, &client).unwrap();
        let plain_grads = backward(&task_loss(&plain), &client).unwrap();

        let captured = p.alias();
        let outs = checkpoint_many_with_client(
            move |xs: &[Var<CpuRuntime>], c: &CpuClient| aux_segment(xs, &captured, c),
            &[&x, &w],
            &client,
        )
        .unwrap();
        let grads = backward(&task_loss(&outs), &client).unwrap();

        let gp = grads.get(p.id()).expect("the aux output must train p");
        assert_eq!(gp.to_vec::<f32>(), [0.9f32, -4.5]);
        for id in [x.id(), w.id(), p.id()] {
            assert_eq!(
                bits(grads.get(id).unwrap(), &client),
                bits(plain_grads.get(id).unwrap(), &client)
            );
        }
    }

    #[test]
    fn test_checkpoint_many_refuses_integer_output() {
        let (device, client) = cpu();
        let x = leaf(&[1.0f32, 2.0], &[2], &device);
        let err = checkpoint_many_with_client(
            move |xs: &[Var<CpuRuntime>], c: &CpuClient| {
                let index = Tensor::<CpuRuntime>::from_slice(&[0i32, 1], &[2], &device)?;
                Ok(vec![var_mul(&xs[0], &xs[0], c)?, Var::new(index, false)])
            },
            &[&x],
            &client,
        )
        .expect_err("an integer output must be refused");
        let text = err.to_string();
        assert!(text.contains("output 1 has dtype I32"), "{text}");
        assert!(text.contains("only float outputs"), "{text}");
    }

    #[test]
    fn test_checkpoint_many_refuses_empty_output_list() {
        let (device, client) = cpu();
        let x = leaf(&[1.0f32], &[1], &device);
        let err = checkpoint_many_with_client(
            |_: &[Var<CpuRuntime>], _: &CpuClient| Ok(Vec::new()),
            &[&x],
            &client,
        )
        .expect_err("an empty output list must be refused");
        assert!(err.to_string().contains("returned no outputs"));
    }

    #[test]
    fn test_checkpoint_many_recompute_layout_change_errors() {
        let (device, client) = cpu();
        let x = leaf(&[1.0f32, 2.0, 3.0], &[3], &device);
        let calls = AtomicUsize::new(0);
        let outs = checkpoint_many_with_client(
            move |xs: &[Var<CpuRuntime>], c: &CpuClient| {
                let square = var_mul(&xs[0], &xs[0], c)?;
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(vec![square])
                } else {
                    Ok(vec![var_sum(&square, &[], false, c)?])
                }
            },
            &[&x],
            &client,
        )
        .unwrap();

        let err = backward(&loss(&outs, &client), &client)
            .err()
            .expect("a recompute with a new layout must fail");
        let text = err.to_string();
        assert!(text.contains("same shapes and dtypes"), "{text}");
        assert!(
            text.contains("[F32 [3]]") && text.contains("[F32 []]"),
            "{text}"
        );
    }
}
