#![allow(dead_code)]

//! CPU pairwise distance benchmarks.
//!
//! Each bench computes `cdist` of one query row against `M` rows of width `d`.

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::ops::{DistanceMetric, DistanceOps};
use numr::prelude::*;

/// Number of rows in the reference set.
const M: usize = 64;

fn run(b: &mut Bencher, d: usize, metric: DistanceMetric) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let x = client.rand(&[1, d], DType::F32).unwrap();
    let y = client.rand(&[M, d], DType::F32).unwrap();
    b.iter(|| black_box(client.cdist(&x, &y, metric).unwrap()));
}

#[flux::bench(group = "cdist_sqeuclidean_f32", args = [128, 384, 768, 1536])]
fn cdist_sqeuclidean(b: &mut Bencher, d: usize) {
    run(b, d, DistanceMetric::SquaredEuclidean);
}

#[flux::bench(group = "cdist_cosine_f32", args = [128, 384, 768, 1536])]
fn cdist_cosine(b: &mut Bencher, d: usize) {
    run(b, d, DistanceMetric::Cosine);
}

#[flux::bench(group = "cdist_manhattan_f32", args = [128, 384, 768, 1536])]
fn cdist_manhattan(b: &mut Bencher, d: usize) {
    run(b, d, DistanceMetric::Manhattan);
}

fn main() {
    fluxbench::run().unwrap();
}
