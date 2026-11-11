use numr::dtype::{DType, Element};
use numr::ops::{BinaryOps, IndexingOps};
use numr::runtime::Runtime;
use numr::tensor::{Layout, Tensor};

fn exact<R: Runtime<DType = DType>, T: Element + bytemuck::Pod>(
    tensor: &Tensor<R>,
    expected: &[T],
) {
    assert_eq!(
        bytemuck::cast_slice::<T, u8>(&tensor.to_vec::<T>()),
        bytemuck::cast_slice::<T, u8>(expected),
    );
}

pub fn contiguous<R, C, T>(client: &C, device: &R::Device, value: impl Fn(i64) -> T)
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
    T: Element + bytemuck::Pod,
{
    let original = [value(1), value(2), value(3), value(4)];
    let a = Tensor::<R>::from_slice(&original, &[4], device).unwrap();
    let alias = a.clone();
    let index = Tensor::<R>::from_slice(&[1i32, 3], &[2], device).unwrap();
    let src = Tensor::<R>::from_slice(&[value(20), value(40)], &[2], device).unwrap();
    let result = client.index_put(&a, 0, &index, &src).unwrap();
    assert_eq!(result.shape(), &[4]);
    exact(&result, &[value(1), value(20), value(3), value(40)]);
    exact(&a, &original);
    exact(&alias, &original);
    exact(&src, &[value(20), value(40)]);
}

pub fn views_and_aliases<R, C, T>(client: &C, device: &R::Device, value: impl Fn(i64) -> T)
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R> + BinaryOps<R>,
    T: Element + bytemuck::Pod,
{
    let original = [value(1), value(2), value(3), value(4)];
    let a = Tensor::<R>::from_slice(&original, &[4], device).unwrap();
    let reverse = Tensor::<R>::from_slice(&[3i32, 2, 1, 0], &[4], device).unwrap();
    let result = client.index_put(&a, 0, &reverse, &a).unwrap();
    exact(&result, &[value(4), value(3), value(2), value(1)]);
    exact(&a, &original);

    let backing_values = [value(90), value(1), value(2), value(3), value(4), value(91)];
    let backing = Tensor::<R>::from_slice(&backing_values, &[6], device).unwrap();
    let middle = backing.narrow(0, 1, 4).unwrap();
    let index = Tensor::<R>::from_slice(&[2i32], &[1], device).unwrap();
    let src = Tensor::<R>::from_slice(&[value(8)], &[1], device).unwrap();
    let result = client.index_put(&middle, 0, &index, &src).unwrap();
    exact(&result, &[value(1), value(2), value(8), value(4)]);
    exact(&backing, &backing_values);

    let a_values = [value(1), value(2), value(3), value(4), value(5), value(6)];
    let src_values = [
        value(10),
        value(20),
        value(30),
        value(40),
        value(50),
        value(60),
    ];
    let a_backing = Tensor::<R>::from_slice(&a_values, &[2, 3], device).unwrap();
    let src_backing = Tensor::<R>::from_slice(&src_values, &[2, 3], device).unwrap();
    let index_backing = Tensor::<R>::from_slice(&[1i32, 7, 0, 7], &[2, 2], device).unwrap();
    let index = Tensor::<R>::from_parts(
        index_backing.storage().clone(),
        Layout::new(vec![2], vec![2isize], 0),
    );
    assert!(!index.is_contiguous());
    let result = client
        .index_put(
            &a_backing.transpose(0, 1).unwrap(),
            1,
            &index,
            &src_backing.transpose(0, 1).unwrap(),
        )
        .unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    exact(
        &result,
        &[
            value(40),
            value(10),
            value(50),
            value(20),
            value(60),
            value(30),
        ],
    );
    exact(&a_backing, &a_values);
    exact(&src_backing, &src_values);
    exact(&index_backing, &[1i32, 7, 0, 7]);

    let empty_index = Tensor::<R>::from_slice(&[] as &[i32], &[0], device).unwrap();
    let empty_src = Tensor::<R>::from_slice(&[] as &[T], &[0], device).unwrap();
    let copied = client.index_put(&a, 0, &empty_index, &empty_src).unwrap();
    exact(&copied, &original);
    let replacement = Tensor::<R>::from_slice(&[value(9); 4], &[4], device).unwrap();
    client.copy_into(&copied, &replacement).unwrap();
    exact(&copied, &[value(9); 4]);
    exact(&a, &original);

    for (shape, dim, src_shape) in [([0, 3], 1, [0, 1]), ([3, 0], 0, [1, 0])] {
        let input = Tensor::<R>::from_slice(&[] as &[T], &shape, device).unwrap();
        let index = Tensor::<R>::from_slice(&[1i32], &[1], device).unwrap();
        let src = Tensor::<R>::from_slice(&[] as &[T], &src_shape, device).unwrap();
        let result = client.index_put(&input, dim, &index, &src).unwrap();
        assert_eq!(result.shape(), &shape);
        exact(&result, &[] as &[T]);
    }
}

pub fn payloads<R, C, T>(client: &C, device: &R::Device, original: &[T], replacement: &[T])
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
    T: Element + bytemuck::Pod,
{
    let a = Tensor::<R>::from_slice(original, &[4], device).unwrap();
    let src = Tensor::<R>::from_slice(replacement, &[2], device).unwrap();
    let index = Tensor::<R>::from_slice(&[1i64, 3], &[2], device).unwrap();
    let result = client.index_put(&a, 0, &index, &src).unwrap();
    exact(
        &result,
        &[original[0], replacement[0], original[2], replacement[1]],
    );
    exact(&a, original);
    exact(&src, replacement);
}

pub fn negative_strides<R, C, T>(client: &C, device: &R::Device, value: impl Fn(i64) -> T)
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
    T: Element + bytemuck::Pod,
{
    let original = [value(1), value(2), value(3), value(4)];
    let source = [value(10), value(20)];
    let a_backing = Tensor::<R>::from_slice(&original, &[4], device).unwrap();
    let index_backing = Tensor::<R>::from_slice(&[0i32, 2], &[2], device).unwrap();
    let src_backing = Tensor::<R>::from_slice(&source, &[2], device).unwrap();
    let a = a_backing.flip(0).unwrap();
    let index = index_backing.flip(0).unwrap();
    let src = src_backing.flip(0).unwrap();
    for strides in [a.strides(), index.strides(), src.strides()] {
        assert_eq!(strides, &[-1]);
    }
    let result = client.index_put(&a, 0, &index, &src).unwrap();
    assert_eq!(result.shape(), &[4]);
    exact(&result, &[value(10), value(3), value(20), value(1)]);
    exact(&a_backing, &original);
    exact(&index_backing, &[0i32, 2]);
    exact(&src_backing, &source);
}
