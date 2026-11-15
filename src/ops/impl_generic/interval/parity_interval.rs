//! Bit parity of native interval kernels against the composed reference.

use super::composed::ComposedClient;
use super::interval_check::IntervalBinary;
use super::interval_impl::{
    interval_binary_impl, interval_capture_impl, interval_flag_bounds_impl,
    interval_flag_nonfinite_impl, interval_flag_nonzero_impl, interval_sqrt_nonnegative_impl,
};
use super::parity_values::{TestFloat, assert_same, flag};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{IntervalEnclosure, IntervalOperand, IntervalOps, IntervalPair};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Every binary operation and enclosure.
pub const CASES: [(IntervalBinary, IntervalEnclosure); 10] = [
    (IntervalBinary::Add, IntervalEnclosure::Outward),
    (IntervalBinary::Add, IntervalEnclosure::Tightened),
    (IntervalBinary::Sub, IntervalEnclosure::Outward),
    (IntervalBinary::Sub, IntervalEnclosure::Tightened),
    (IntervalBinary::Mul, IntervalEnclosure::Outward),
    (IntervalBinary::Mul, IntervalEnclosure::Tightened),
    (IntervalBinary::DivPositive, IntervalEnclosure::Outward),
    (IntervalBinary::DivPositive, IntervalEnclosure::Tightened),
    (
        IntervalBinary::DivPositiveInterval,
        IntervalEnclosure::Outward,
    ),
    (
        IntervalBinary::DivPositiveInterval,
        IntervalEnclosure::Tightened,
    ),
];

/// Every ordered and inverted endpoint pair of the special values.
pub fn interval_lanes<T: TestFloat>() -> (Vec<T>, Vec<T>) {
    let values = T::specials();
    let lower = values
        .iter()
        .flat_map(|&lo| values.iter().map(move |_| lo))
        .collect();
    let upper = values.iter().flat_map(|_| values.iter().copied()).collect();
    (lower, upper)
}

fn native<R, C>(
    client: &C,
    op: IntervalBinary,
    a: IntervalOperand<'_, R>,
    b: IntervalOperand<'_, R>,
    enclosure: IntervalEnclosure,
    status: &Tensor<R>,
) -> Result<IntervalPair<R>>
where
    R: Runtime<DType = DType>,
    C: IntervalOps<R>,
{
    match op {
        IntervalBinary::Add => client.interval_add(a, b, enclosure, status),
        IntervalBinary::Sub => client.interval_sub(a, b, enclosure, status),
        IntervalBinary::Mul => client.interval_mul(a, b, enclosure, status),
        IntervalBinary::DivPositive => client.interval_div_positive(a, b.lower, enclosure, status),
        IntervalBinary::DivPositiveInterval => {
            client.interval_div_positive_interval(a, b, enclosure, status)
        }
    }
}

/// Run one call natively and composed, then compare bits, errors, and status flags.
pub fn compare_call<R, C, T>(
    client: &C,
    op: IntervalBinary,
    a: IntervalOperand<'_, R>,
    b: IntervalOperand<'_, R>,
    enclosure: IntervalEnclosure,
    label: &str,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: IntervalOps<R> + ComposedClient<R>,
    T: TestFloat,
{
    let b = match op {
        IntervalBinary::DivPositive => IntervalOperand {
            lower: b.lower,
            upper: b.lower,
        },
        _ => b,
    };
    let (native_status, reference_status) = (client.interval_status()?, client.interval_status()?);
    let native_result = native(client, op, a, b, enclosure, &native_status);
    let reference_result = interval_binary_impl(client, op, a, b, enclosure, &reference_status);
    match (native_result, reference_result) {
        (Ok(n), Ok(r)) => {
            assert_same::<R, T>(&n.0, &r.0, &format!("{label} lower"))?;
            assert_same::<R, T>(&n.1, &r.1, &format!("{label} upper"))?;
        }
        (Err(n), Err(r)) => assert_eq!(n.to_string(), r.to_string(), "{label} error"),
        (n, r) => panic!("{label}: native {:?} reference {:?}", n.err(), r.err()),
    }
    assert_eq!(
        flag(&native_status)?,
        flag(&reference_status)?,
        "{label} status"
    );
    Ok(())
}

fn tensor<R: Runtime<DType = DType>, T: TestFloat>(
    values: &[T],
    shape: &[usize],
    device: &R::Device,
) -> Result<Tensor<R>> {
    Tensor::from_slice(values, shape, device)
}

/// All lane pairs as contiguous operands, plus a broadcast grid over a prefix.
pub fn binary_grid<R, C, T>(client: &C, device: &R::Device) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: IntervalOps<R> + ComposedClient<R>,
    T: TestFloat,
{
    let (lower, upper) = interval_lanes::<T>();
    let n = lower.len();
    let mut columns: [Vec<T>; 4] = Default::default();
    for i in 0..n {
        for j in 0..n {
            columns[0].push(lower[i]);
            columns[1].push(upper[i]);
            columns[2].push(lower[j]);
            columns[3].push(upper[j]);
        }
    }
    let flat: Vec<Tensor<R>> = columns
        .iter()
        .map(|values| tensor(values, &[n * n], device))
        .collect::<Result<_>>()?;
    let prefix = 48.min(n);
    let column = |values: &[T]| tensor(&values[..prefix], &[prefix, 1], device);
    let row = |values: &[T]| tensor(&values[..prefix], &[1, prefix], device);
    let grid = [column(&lower)?, column(&upper)?, row(&lower)?, row(&upper)?];
    for (op, enclosure) in CASES {
        for (name, operands) in [("flat", &flat[..]), ("broadcast", &grid[..])] {
            let a = IntervalOperand {
                lower: &operands[0],
                upper: &operands[1],
            };
            let b = IntervalOperand {
                lower: &operands[2],
                upper: &operands[3],
            };
            compare_call::<R, C, T>(
                client,
                op,
                a,
                b,
                enclosure,
                &format!("{op:?} {enclosure:?} {name}"),
            )?;
        }
    }
    Ok(())
}

/// Single-lane calls, so each lane's status flag is compared on its own.
pub fn binary_flags<R, C, T>(client: &C, device: &R::Device, stride: usize) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: IntervalOps<R> + ComposedClient<R>,
    T: TestFloat,
{
    let (lower, upper) = interval_lanes::<T>();
    let scalar = |value: T| tensor::<R, T>(&[value], &[], device);
    for (op, enclosure) in CASES {
        for i in (0..lower.len()).step_by(stride) {
            for j in (0..lower.len()).step_by(stride * 3 + 1) {
                let operands = [
                    scalar(lower[i])?,
                    scalar(upper[i])?,
                    scalar(lower[j])?,
                    scalar(upper[j])?,
                ];
                let a = IntervalOperand {
                    lower: &operands[0],
                    upper: &operands[1],
                };
                let b = IntervalOperand {
                    lower: &operands[2],
                    upper: &operands[3],
                };
                compare_call::<R, C, T>(
                    client,
                    op,
                    a,
                    b,
                    enclosure,
                    &format!("{op:?} {enclosure:?} {i} {j}"),
                )?;
            }
        }
    }
    Ok(())
}

/// Empty outputs and broadcast errors keep operand-shaped division flags.
pub fn binary_edges<R, C, T>(client: &C, device: &R::Device) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: IntervalOps<R> + ComposedClient<R>,
    T: TestFloat,
{
    let specials = T::specials();
    let invalid = tensor::<R, T>(&specials[..2], &[2], device)?;
    let empty = tensor::<R, T>(&[], &[0, 2], device)?;
    let three = tensor::<R, T>(&specials[..3], &[3], device)?;
    for (op, enclosure) in CASES {
        for (left, right, name) in [
            (&empty, &invalid, "empty"),
            (&three, &invalid, "mismatch"),
            (&invalid, &three, "swapped"),
        ] {
            let a = IntervalOperand {
                lower: left,
                upper: left,
            };
            let b = IntervalOperand {
                lower: right,
                upper: right,
            };
            compare_call::<R, C, T>(
                client,
                op,
                a,
                b,
                enclosure,
                &format!("{op:?} {enclosure:?} {name}"),
            )?;
        }
    }
    Ok(())
}

/// Square roots and the three status flag operations, lane by lane.
pub fn unary_ops<R, C, T>(client: &C, device: &R::Device) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: IntervalOps<R> + ComposedClient<R>,
    T: TestFloat,
{
    let (lower, upper) = interval_lanes::<T>();
    let n = lower.len();
    let (lo, hi) = (
        tensor::<R, T>(&lower, &[n], device)?,
        tensor::<R, T>(&upper, &[n], device)?,
    );
    let whole = IntervalOperand {
        lower: &lo,
        upper: &hi,
    };
    let (s1, s2) = (client.interval_status()?, client.interval_status()?);
    let native = client.interval_sqrt_nonnegative(whole, &s1)?;
    let reference = interval_sqrt_nonnegative_impl(client, whole, &s2)?;
    assert_same::<R, T>(&native.0, &reference.0, "sqrt lower")?;
    assert_same::<R, T>(&native.1, &reference.1, "sqrt upper")?;
    for (name, value) in [
        ("pair", whole),
        (
            "same view",
            IntervalOperand {
                lower: &lo,
                upper: &lo,
            },
        ),
    ] {
        let (s1, s2) = (client.interval_status()?, client.interval_status()?);
        let native = client.interval_capture(value, &s1)?;
        let reference = interval_capture_impl(client, value, &s2)?;
        assert_same::<R, T>(&native.0, &reference.0, &format!("capture {name} lower"))?;
        assert_same::<R, T>(&native.1, &reference.1, &format!("capture {name} upper"))?;
        assert_eq!(flag(&s1)?, flag(&s2)?, "capture {name} status");
    }
    for i in 0..n {
        let (l, h) = (
            tensor::<R, T>(&[lower[i]], &[1], device)?,
            tensor::<R, T>(&[upper[i]], &[1], device)?,
        );
        let lane = IntervalOperand {
            lower: &l,
            upper: &h,
        };
        let statuses: Vec<Tensor<R>> = (0..10)
            .map(|_| client.interval_status())
            .collect::<Result<_>>()?;
        client.interval_capture(lane, &statuses[8])?;
        interval_capture_impl(client, lane, &statuses[9])?;
        client.interval_sqrt_nonnegative(lane, &statuses[0])?;
        interval_sqrt_nonnegative_impl(client, lane, &statuses[1])?;
        client.interval_flag_bounds(lane, &statuses[2])?;
        interval_flag_bounds_impl(client, lane, &statuses[3])?;
        client.interval_flag_nonfinite(&l, &statuses[4])?;
        interval_flag_nonfinite_impl(client, &l, &statuses[5])?;
        client.interval_flag_nonzero(&l, &statuses[6])?;
        interval_flag_nonzero_impl(client, &l, &statuses[7])?;
        for pair in statuses.chunks(2) {
            assert_eq!(flag(&pair[0])?, flag(&pair[1])?, "unary lane {i}");
        }
    }
    Ok(())
}
