#![allow(dead_code)]

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (device, client)
}

fn rand_t(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

fn rand_indices(n: usize, max_val: i32, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let data: Vec<i32> = (0..n).map(|i| (i as i32) % max_val).collect();
    Tensor::<CpuRuntime>::from_slice(&data, &[n], device)
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

#[flux::bench(group = "gather_f32")]
fn numr_gather_1k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1000, 64], &device);
    let idx = rand_indices(500, 1000, &device);
    let idx = idx.reshape(&[500, 1]).unwrap();
    let idx = {
        let client = CpuRuntime::default_client(&device);
        client.repeat(&idx, &[1, 64]).unwrap()
    };
    b.iter(|| black_box(client.gather(&t, 0, &idx).unwrap()));
}

#[flux::bench(group = "gather_f32")]
fn numr_gather_100k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000, 64], &device);
    let idx = rand_indices(10_000, 100_000, &device);
    let idx = idx.reshape(&[10_000, 1]).unwrap();
    let idx = {
        let client = CpuRuntime::default_client(&device);
        client.repeat(&idx, &[1, 64]).unwrap()
    };
    b.iter(|| black_box(client.gather(&t, 0, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// index_select
// ---------------------------------------------------------------------------

#[flux::bench(group = "index_select_f32")]
fn numr_index_select_1k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1000, 128], &device);
    let idx = rand_indices(256, 1000, &device);
    b.iter(|| black_box(client.index_select(&t, 0, &idx).unwrap()));
}

#[flux::bench(group = "index_select_f32")]
fn numr_index_select_100k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000, 128], &device);
    let idx = rand_indices(10_000, 100_000, &device);
    b.iter(|| black_box(client.index_select(&t, 0, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// take (flat indexing)
// ---------------------------------------------------------------------------

#[flux::bench(group = "take_f32")]
fn numr_take_10k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000], &device);
    let idx = rand_indices(10_000, 100_000, &device);
    b.iter(|| black_box(client.take(&t, &idx).unwrap()));
}

#[flux::bench(group = "take_f32")]
fn numr_take_100k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1_000_000], &device);
    let idx = rand_indices(100_000, 1_000_000, &device);
    b.iter(|| black_box(client.take(&t, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// scatter
// ---------------------------------------------------------------------------

#[flux::bench(group = "scatter_f32")]
fn numr_scatter_1k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1000, 64], &device);
    let src = rand_t(&[500, 64], &device);
    let idx = rand_indices(500, 1000, &device);
    let idx = idx.reshape(&[500, 1]).unwrap();
    let idx = {
        let c = CpuRuntime::default_client(&device);
        c.repeat(&idx, &[1, 64]).unwrap()
    };
    b.iter(|| black_box(client.scatter(&t, 0, &idx, &src).unwrap()));
}

// ---------------------------------------------------------------------------
// put (flat scatter)
// ---------------------------------------------------------------------------

#[flux::bench(group = "put_f32")]
fn numr_put_10k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000], &device);
    let idx = rand_indices(10_000, 100_000, &device);
    let vals = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.put(&t, &idx, &vals).unwrap()));
}

// ---------------------------------------------------------------------------
// embedding_lookup (common ML pattern)
// ---------------------------------------------------------------------------

#[flux::bench(group = "embedding_f32")]
fn numr_embedding_32k_vocab(b: &mut Bencher) {
    let (device, client) = setup();
    let embeddings = rand_t(&[32_000, 128], &device);
    let idx = rand_indices(512, 32_000, &device);
    b.iter(|| black_box(client.embedding_lookup(&embeddings, &idx).unwrap()));
}

#[flux::bench(group = "embedding_f32")]
fn numr_embedding_128k_vocab(b: &mut Bencher) {
    let (device, client) = setup();
    let embeddings = rand_t(&[128_000, 128], &device);
    let idx = rand_indices(512, 128_000, &device);
    b.iter(|| black_box(client.embedding_lookup(&embeddings, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[flux::compare(
    id = "index_select_cmp",
    title = "index_select: 1K vs 100K source rows",
    benchmarks = ["numr_index_select_1k", "numr_index_select_100k"],
    baseline = "numr_index_select_1k",
    metric = "mean"
)]
struct IndexSelectCmp;

#[flux::compare(
    id = "take_cmp",
    title = "take: 10K vs 100K indices",
    benchmarks = ["numr_take_10k", "numr_take_100k"],
    baseline = "numr_take_10k",
    metric = "mean"
)]
struct TakeCmp;

#[flux::compare(
    id = "embedding_cmp",
    title = "Embedding: 32K vs 128K vocab",
    benchmarks = ["numr_embedding_32k_vocab", "numr_embedding_128k_vocab"],
    baseline = "numr_embedding_32k_vocab",
    metric = "mean"
)]
struct EmbeddingCmp;

fn main() {
    fluxbench_cli::run().unwrap();
}
