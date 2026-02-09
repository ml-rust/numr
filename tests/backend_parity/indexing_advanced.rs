// Backend parity tests for advanced indexing operations

use numr::ops::{IndexingOps, ScatterReduceOp};
use numr::tensor::Tensor;

use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

#[test]
fn test_index_select_parity() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = [2i64, 0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_x = Tensor::from_slice(&input, &[3, 2], &cpu_device);
    let cpu_i = Tensor::from_slice(&indices, &[2], &cpu_device);
    let cpu: Vec<f32> = cpu_client.index_select(&cpu_x, 0, &cpu_i).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let x = Tensor::from_slice(&input, &[3, 2], &cuda_device);
        let i = Tensor::from_slice(&indices, &[2], &cuda_device);
        let got: Vec<f32> = cuda_client.index_select(&x, 0, &i).unwrap().to_vec();
        assert_parity_f32(&cpu, &got, "index_select_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = Tensor::from_slice(&input, &[3, 2], &wgpu_device);
        let i = Tensor::from_slice(&indices, &[2], &wgpu_device);
        let got: Vec<f32> = wgpu_client.index_select(&x, 0, &i).unwrap().to_vec();
        assert_parity_f32(&cpu, &got, "index_select_wgpu");
    });
}

#[test]
fn test_gather_scatter_parity() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gather_indices = [0i64, 2, 1, 0];
    let src = [1.0f32, 2.0, 3.0, 4.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_x = Tensor::from_slice(&input, &[3, 2], &cpu_device);
    let cpu_i = Tensor::from_slice(&gather_indices, &[2, 2], &cpu_device);
    let cpu_g: Vec<f32> = cpu_client.gather(&cpu_x, 0, &cpu_i).unwrap().to_vec();

    let cpu_dst = Tensor::from_slice(&[0.0f32; 6], &[3, 2], &cpu_device);
    let cpu_src = Tensor::from_slice(&src, &[2, 2], &cpu_device);
    let cpu_s: Vec<f32> = cpu_client
        .scatter(&cpu_dst, 0, &cpu_i, &cpu_src)
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let x = Tensor::from_slice(&input, &[3, 2], &cuda_device);
        let i = Tensor::from_slice(&gather_indices, &[2, 2], &cuda_device);
        let g: Vec<f32> = cuda_client.gather(&x, 0, &i).unwrap().to_vec();
        assert_parity_f32(&cpu_g, &g, "gather_cuda");

        let dst = Tensor::from_slice(&[0.0f32; 6], &[3, 2], &cuda_device);
        let src_t = Tensor::from_slice(&src, &[2, 2], &cuda_device);
        let s: Vec<f32> = cuda_client.scatter(&dst, 0, &i, &src_t).unwrap().to_vec();
        assert_parity_f32(&cpu_s, &s, "scatter_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = Tensor::from_slice(&input, &[3, 2], &wgpu_device);
        let i = Tensor::from_slice(&gather_indices, &[2, 2], &wgpu_device);
        let g: Vec<f32> = wgpu_client.gather(&x, 0, &i).unwrap().to_vec();
        assert_parity_f32(&cpu_g, &g, "gather_wgpu");

        let dst = Tensor::from_slice(&[0.0f32; 6], &[3, 2], &wgpu_device);
        let src_t = Tensor::from_slice(&src, &[2, 2], &wgpu_device);
        let s: Vec<f32> = wgpu_client.scatter(&dst, 0, &i, &src_t).unwrap().to_vec();
        assert_parity_f32(&cpu_s, &s, "scatter_wgpu");
    });
}

#[test]
fn test_gather_nd_bincount_embedding_parity() {
    let (cpu_client, cpu_device) = create_cpu_client();

    let input = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[2, 2], &cpu_device);
    let nd_idx = Tensor::from_slice(&[0i64, 0, 1, 1], &[2, 2], &cpu_device);
    let cpu_nd: Vec<f32> = cpu_client.gather_nd(&input, &nd_idx).unwrap().to_vec();

    let bins_input = Tensor::from_slice(&[0i64, 1, 1, 3, 2, 1, 3], &[7], &cpu_device);
    let cpu_bins: Vec<i64> = cpu_client.bincount(&bins_input, None, 0).unwrap().to_vec();

    let emb = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[4, 2],
        &cpu_device,
    );
    let emb_idx = Tensor::from_slice(&[3i64, 0, 1], &[3], &cpu_device);
    let cpu_emb: Vec<f32> = cpu_client
        .embedding_lookup(&emb, &emb_idx)
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[2, 2], &cuda_device);
        let i = Tensor::from_slice(&[0i64, 0, 1, 1], &[2, 2], &cuda_device);
        let nd: Vec<f32> = cuda_client.gather_nd(&x, &i).unwrap().to_vec();
        assert_parity_f32(&cpu_nd, &nd, "gather_nd_cuda");

        let b_in = Tensor::from_slice(&[0i64, 1, 1, 3, 2, 1, 3], &[7], &cuda_device);
        let bins: Vec<i64> = cuda_client.bincount(&b_in, None, 0).unwrap().to_vec();
        assert_eq!(cpu_bins, bins);

        let e = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4, 2],
            &cuda_device,
        );
        let ei = Tensor::from_slice(&[3i64, 0, 1], &[3], &cuda_device);
        let emb_out: Vec<f32> = cuda_client.embedding_lookup(&e, &ei).unwrap().to_vec();
        assert_parity_f32(&cpu_emb, &emb_out, "embedding_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[2, 2], &wgpu_device);
        let i = Tensor::from_slice(&[0i64, 0, 1, 1], &[2, 2], &wgpu_device);
        let nd: Vec<f32> = wgpu_client.gather_nd(&x, &i).unwrap().to_vec();
        assert_parity_f32(&cpu_nd, &nd, "gather_nd_wgpu");

        let b_in = Tensor::from_slice(&[0i64, 1, 1, 3, 2, 1, 3], &[7], &wgpu_device);
        let bins: Vec<i64> = wgpu_client.bincount(&b_in, None, 0).unwrap().to_vec();
        assert_eq!(cpu_bins, bins);

        let e = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4, 2],
            &wgpu_device,
        );
        let ei = Tensor::from_slice(&[3i64, 0, 1], &[3], &wgpu_device);
        let emb_out: Vec<f32> = wgpu_client.embedding_lookup(&e, &ei).unwrap().to_vec();
        assert_parity_f32(&cpu_emb, &emb_out, "embedding_wgpu");
    });
}

#[test]
fn test_scatter_reduce_sum_parity() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let dst = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &cpu_device);
    let idx = Tensor::from_slice(&[0i64, 0, 2], &[3], &cpu_device);
    let src = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &cpu_device);
    let cpu: Vec<f32> = cpu_client
        .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Sum, false)
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let d = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &cuda_device);
        let i = Tensor::from_slice(&[0i64, 0, 2], &[3], &cuda_device);
        let s = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &cuda_device);
        let got: Vec<f32> = cuda_client
            .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Sum, false)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "scatter_reduce_sum_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let d = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &wgpu_device);
        let i = Tensor::from_slice(&[0i64, 0, 2], &[3], &wgpu_device);
        let s = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &wgpu_device);
        let got: Vec<f32> = wgpu_client
            .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Sum, false)
            .unwrap()
            .to_vec();
        assert_parity_f32(&cpu, &got, "scatter_reduce_sum_wgpu");
    });
}
