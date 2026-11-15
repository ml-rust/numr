//! Bit parity of native expansion kernels against the composed reference.

use super::composed::ComposedClient;
use super::expansion_check::ExpansionBinary;
use super::expansion_impl::{
    expansion_binary_impl, expansion_flag_impl, expansion_from_interval_impl,
    expansion_interval_impl, expansion_normalize_impl, expansion_point_impl,
};
use super::parity_values::{Lcg, TestFloat, assert_same, assert_same_nan_class, flag};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ExpansionOperand, ExpansionOps, ExpansionTriple, IntervalOperand};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Random component lanes drawn from the special values and nearby ordinary values.
fn components<T: TestFloat>(lanes: usize, seed: u64) -> [Vec<T>; 3] {
    let specials = T::specials();
    let ordinary: Vec<T> = (1..=12)
        .map(|k| T::from_f64(1.0 / f64::from(k) + 0.375 * f64::from(k)))
        .collect();
    let pool: Vec<T> = specials.iter().chain(&ordinary).copied().collect();
    let mut rng = Lcg(seed);
    let mut draw = |small: bool| {
        let value = pool[rng.below(pool.len())];
        // Mostly small, nonnegative radii and low components keep most lanes valid.
        if small && rng.below(3) != 0 {
            T::from_f64(value.to_f64().abs() * 2f64.powi(-60))
        } else {
            value
        }
    };
    let mut out: [Vec<T>; 3] = Default::default();
    for _ in 0..lanes {
        out[0].push(draw(false));
        out[1].push(draw(true));
        out[2].push(draw(true));
    }
    out
}

fn upload<R: Runtime<DType = DType>, T: TestFloat>(
    parts: &[Vec<T>; 3],
    device: &R::Device,
) -> Result<[Tensor<R>; 3]> {
    let n = parts[0].len();
    Ok([
        Tensor::from_slice(&parts[0], &[n], device)?,
        Tensor::from_slice(&parts[1], &[n], device)?,
        Tensor::from_slice(&parts[2], &[n], device)?,
    ])
}

fn view<R: Runtime<DType = DType>>(parts: &[Tensor<R>; 3]) -> ExpansionOperand<'_, R> {
    ExpansionOperand {
        high: &parts[0],
        low: &parts[1],
        abs_error: &parts[2],
    }
}

/// Compare components. `fused` lanes passed NaNs through fused multiply-add.
fn same_triple<R: Runtime<DType = DType>, T: TestFloat>(
    native: &ExpansionTriple<R>,
    reference: &ExpansionTriple<R>,
    label: &str,
    fused: bool,
) -> Result<()> {
    let check = if fused {
        assert_same_nan_class::<R, T>
    } else {
        assert_same::<R, T>
    };
    check(&native.0, &reference.0, &format!("{label} high"))?;
    check(&native.1, &reference.1, &format!("{label} low"))?;
    check(&native.2, &reference.2, &format!("{label} error"))
}

/// Compare every expansion operation on `lanes` lanes, and status flags on single lanes.
pub fn expansion_ops<R, C, T>(
    client: &C,
    device: &R::Device,
    lanes: usize,
    single: usize,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: ExpansionOps<R> + crate::ops::IntervalOps<R> + ComposedClient<R>,
    T: TestFloat,
{
    for (count, seed, single_lanes) in [(lanes, 7, false), (single, 11, true)] {
        let chunks: Vec<usize> = if single_lanes {
            (0..count).collect()
        } else {
            vec![0]
        };
        let a_all = components::<T>(count, seed);
        let b_all = components::<T>(count, seed + 1);
        for start in chunks {
            let width = if single_lanes { 1 } else { count };
            let slice = |parts: &[Vec<T>; 3]| -> [Vec<T>; 3] {
                [0, 1, 2].map(|k| parts[k][start..start + width].to_vec())
            };
            let (a, b) = (
                upload::<R, T>(&slice(&a_all), device)?,
                upload::<R, T>(&slice(&b_all), device)?,
            );
            let (a, b) = (view(&a), view(&b));
            let label = format!("lane {start}");
            for op in [
                ExpansionBinary::Add,
                ExpansionBinary::Sub,
                ExpansionBinary::Mul,
                ExpansionBinary::DivPositive,
            ] {
                let (s1, s2) = (client.interval_status()?, client.interval_status()?);
                let native = match op {
                    ExpansionBinary::Add => client.expansion_add(a, b, &s1)?,
                    ExpansionBinary::Sub => client.expansion_sub(a, b, &s1)?,
                    ExpansionBinary::Mul => client.expansion_mul(a, b, &s1)?,
                    ExpansionBinary::DivPositive => client.expansion_div_positive(a, b, &s1)?,
                };
                let reference = expansion_binary_impl(client, op, a, b, &s2)?;
                same_triple::<R, T>(
                    &native,
                    &reference,
                    &format!("{op:?} {label}"),
                    op.needs_fused(),
                )?;
                assert_eq!(flag(&s1)?, flag(&s2)?, "{op:?} {label} status");
            }
            let lift = IntervalOperand {
                lower: a.high,
                upper: b.high,
            };
            let (s1, s2) = (client.interval_status()?, client.interval_status()?);
            let native = client.expansion_from_interval(lift, &s1)?;
            let reference = expansion_from_interval_impl(client, lift, &s2)?;
            same_triple::<R, T>(&native, &reference, &format!("lift {label}"), false)?;
            assert_eq!(flag(&s1)?, flag(&s2)?, "lift {label} status");
            let statuses: Vec<Tensor<R>> = (0..8)
                .map(|_| client.interval_status())
                .collect::<Result<_>>()?;
            let native = client.expansion_interval(a, &statuses[0])?;
            let reference = expansion_interval_impl(client, a, &statuses[1])?;
            assert_same::<R, T>(&native.0, &reference.0, &format!("interval lower {label}"))?;
            assert_same::<R, T>(&native.1, &reference.1, &format!("interval upper {label}"))?;
            let native = client.expansion_normalize(a, &statuses[2])?;
            let reference = expansion_normalize_impl(client, a, &statuses[3])?;
            same_triple::<R, T>(&native, &reference, &format!("normalize {label}"), false)?;
            let native = client.expansion_point(a.high, &statuses[4])?;
            let reference = expansion_point_impl(client, a.high, &statuses[5])?;
            same_triple::<R, T>(&native, &reference, &format!("point {label}"), false)?;
            client.expansion_flag(a, &statuses[6])?;
            expansion_flag_impl(client, a, &statuses[7])?;
            for (k, pair) in statuses.chunks(2).enumerate() {
                assert_eq!(flag(&pair[0])?, flag(&pair[1])?, "unary {k} {label} status");
            }
        }
    }
    Ok(())
}
