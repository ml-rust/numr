#![allow(dead_code)]

use fluxbench::{Bencher, flux};
use numr::prelude::*;
use std::hint::black_box;

#[flux::bench]
fn numr_256(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = client.rand(&[256, 256], DType::F32).unwrap();
    let bm = client.rand(&[256, 256], DType::F32).unwrap();
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench]
fn numr_512(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = client.rand(&[512, 512], DType::F32).unwrap();
    let bm = client.rand(&[512, 512], DType::F32).unwrap();
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

fn main() {
    fluxbench_cli::run().unwrap();
}
