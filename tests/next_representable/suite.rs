//! Exact-bit checks shared by public API and actual backend tests.

use super::fixtures::*;
use numr::prelude::*;

fn metadata<R: Runtime<DType = DType>>(input: &Tensor<R>, output: &Tensor<R>) {
    assert_eq!(output.shape(), input.shape());
    assert_eq!(output.dtype(), input.dtype());
    assert!(output.device().is_same(input.device()));
    if !input.is_empty() {
        assert!(!output.shares_storage_with(input));
    }
}

macro_rules! suite {
    ($suite:ident, $check:ident, $read:ident, $float:ty, $bits:ty, $dtype:expr, $fixture:ident, $reference:ident) => {
        fn $read<R: Runtime>(tensor: &Tensor<R>) -> Result<Vec<$bits>> {
            Ok(tensor
                .to_bytes()?
                .chunks_exact(std::mem::size_of::<$bits>())
                .map(|chunk| {
                    let mut bytes = [0; std::mem::size_of::<$bits>()];
                    bytes.copy_from_slice(chunk);
                    <$bits>::from_ne_bytes(bytes)
                })
                .collect())
        }

        fn $check<R, C>(client: &C, input: &Tensor<R>, original: &[$bits]) -> Result<Vec<Vec<u8>>>
        where
            R: Runtime<DType = DType>,
            C: NextRepresentableOps<R>,
        {
            let mut outputs = Vec::new();
            for up in [true, false] {
                let output = if up {
                    client.next_up(input)?
                } else {
                    client.next_down(input)?
                };
                metadata(input, &output);
                let actual: Vec<_> = $read(&output)?;
                let expected: Vec<_> = original.iter().map(|&bits| $reference(bits, up)).collect();
                assert_eq!(
                    actual,
                    expected,
                    "{} {}",
                    stringify!($float),
                    if up { "next_up" } else { "next_down" }
                );
                outputs.push(actual.into_iter().flat_map(<$bits>::to_ne_bytes).collect());
            }
            Ok(outputs)
        }

        pub fn $suite<R, C>(client: &C, device: &R::Device) -> Result<Vec<Vec<u8>>>
        where
            R: Runtime<DType = DType>,
            C: NextRepresentableOps<R>,
        {
            let bits = $fixture();
            let values: Vec<_> = bits.iter().copied().map(<$float>::from_bits).collect();
            let input = Tensor::<R>::from_slice(&values, &[values.len()], device)?;
            let mut outputs = $check(client, &input, &bits)?;
            assert_eq!($read(&input)?, bits);
            for len in [
                0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
            ] {
                let input = Tensor::<R>::from_slice(&values[..len], &[len], device)?;
                outputs.extend($check(client, &input, &bits[..len])?);
            }
            let scalar = Tensor::<R>::from_slice(&values[..1], &[], device)?;
            outputs.extend($check(client, &scalar, &bits[..1])?);
            let empty = Tensor::<R>::empty(&[2, 0, 3], $dtype, device)?;
            outputs.extend($check(client, &empty, &[])?);

            let layout_bits = [
                bits[0], bits[1], bits[2], bits[3], bits[16], bits[17], bits[18], bits[19],
                bits[20], bits[21], bits[26], bits[27],
            ];
            let layout_values: Vec<_> = layout_bits
                .iter()
                .copied()
                .map(<$float>::from_bits)
                .collect();
            let storage = Tensor::<R>::from_slice(&layout_values, &[3, 4], device)?;
            let offset = storage.narrow(0, 1, 2)?;
            assert!(offset.offset() > 0 && offset.is_contiguous());
            outputs.extend($check(client, &offset, &layout_bits[4..12])?);
            let transpose = storage.transpose(0, 1)?;
            let bits_ref = &layout_bits;
            let transposed: Vec<_> = (0..4)
                .flat_map(|column| (0..3).map(move |row| bits_ref[row * 4 + column]))
                .collect();
            outputs.extend($check(client, &transpose, &transposed)?);
            let slice = storage.narrow(1, 1, 2)?;
            let sliced = [
                layout_bits[1],
                layout_bits[2],
                layout_bits[5],
                layout_bits[6],
                layout_bits[9],
                layout_bits[10],
            ];
            outputs.extend($check(client, &slice, &sliced)?);
            let flipped = storage.flip(1)?;
            let flipped_bits: Vec<_> = (0..3)
                .flat_map(|row| (0..4).rev().map(move |column| bits_ref[row * 4 + column]))
                .collect();
            outputs.extend($check(client, &flipped, &flipped_bits)?);
            let row = Tensor::<R>::from_slice(&values[20..24], &[1, 4], device)?;
            let broadcast = row.broadcast_to(&[3, 4])?;
            let broadcast_bits: Vec<_> =
                (0..3).flat_map(|_| bits[20..24].iter().copied()).collect();
            outputs.extend($check(client, &broadcast, &broadcast_bits)?);

            assert_eq!($read(&storage)?, layout_bits);
            assert_eq!($read(&row)?, bits[20..24]);

            for up in [true, false] {
                let input = Tensor::<R>::from_slice(&values[..4], &[4], device)?;
                let output = if up {
                    client.next_up(&input)?
                } else {
                    client.next_down(&input)?
                };
                let snapshot = $read(&output)?;
                let replacement = (7 as $float).to_bits().to_ne_bytes().repeat(4);
                R::copy_to_device(&replacement, input.ptr(), device)?;
                assert_eq!($read(&output)?, snapshot);
                let replacement = (9 as $float).to_bits().to_ne_bytes().repeat(4);
                R::copy_to_device(&replacement, output.ptr(), device)?;
                assert_eq!(input.try_to_vec::<$float>()?, vec![7 as $float; 4]);
            }
            Ok(outputs)
        }
    };
}

suite!(
    f32_suite,
    check_f32,
    read_f32,
    f32,
    u32,
    DType::F32,
    f32_bits,
    reference_f32
);
suite!(
    f64_suite,
    check_f64,
    read_f64,
    f64,
    u64,
    DType::F64,
    f64_bits,
    reference_f64
);

pub fn unsupported<R, C>(client: &C, device: &R::Device, reject_f64: bool) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: NextRepresentableOps<R>,
{
    let mut dtypes = vec![
        DType::F16,
        DType::BF16,
        DType::FP8E4M3,
        DType::FP8E5M2,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool,
        DType::Complex64,
        DType::Complex128,
    ];
    if reject_f64 {
        dtypes.push(DType::F64);
    }
    for dtype in dtypes {
        for shape in [&[0, 3][..], &[1][..]] {
            let input = Tensor::<R>::empty(shape, dtype, device)?;
            for (op, result) in [
                ("next_up", client.next_up(&input)),
                ("next_down", client.next_down(&input)),
            ] {
                assert!(
                    matches!(result, Err(Error::UnsupportedDType { dtype: actual, op: actual_op }) if actual == dtype && actual_op == op),
                    "{op} {dtype:?} {shape:?}"
                );
            }
        }
    }
    Ok(())
}
