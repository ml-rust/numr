// WebGPU multinomial without replacement.
//
// The shader runs one workgroup per distribution and reads its row from the
// workgroup id. The sampler is random, so the checks are the properties that
// hold on any backend: shape, index range, no repeat within a row, zero
// probability never sampled, and a full draw returning exactly the nonzero set.
//
// Every test is `#[cfg(feature = "wgpu")]`, so the imports are too.
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "wgpu")]
use numr::ops::RandomOps;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
#[cfg(feature = "wgpu")]
use numr::tensor::Tensor;

/// Check every property of a without-replacement draw, row by row.
#[cfg(feature = "wgpu")]
fn check_draw(indices: &[i32], probs: &[f32], rows: usize, cats: usize, samples: usize) {
    assert_eq!(indices.len(), rows * samples, "output element count");
    for r in 0..rows {
        let row = &indices[r * samples..(r + 1) * samples];
        let row_probs = &probs[r * cats..(r + 1) * cats];
        let mut seen = vec![false; cats];
        for &idx in row {
            assert!(
                idx >= 0 && (idx as usize) < cats,
                "row {r}: index {idx} outside 0..{cats}"
            );
            let i = idx as usize;
            assert!(!seen[i], "row {r}: index {i} repeated in {row:?}");
            assert!(
                row_probs[i] > 0.0,
                "row {r}: zero-probability category {i} sampled"
            );
            seen[i] = true;
        }
        let nonzero = row_probs.iter().filter(|&&p| p > 0.0).count();
        if samples == nonzero {
            for (i, &p) in row_probs.iter().enumerate() {
                assert_eq!(seen[i], p > 0.0, "row {r}: full draw missed category {i}");
            }
        }
    }
}

/// Two zeros per row, at positions that move from row to row.
#[cfg(feature = "wgpu")]
fn striped_probs(rows: usize, cats: usize) -> Vec<f32> {
    let mut probs = Vec::with_capacity(rows * cats);
    for r in 0..rows {
        for c in 0..cats {
            if c == r % cats || c == (r + 3) % cats {
                probs.push(0.0);
            } else {
                probs.push(0.1 * (1 + (r * 7 + c * 3) % 5) as f32);
            }
        }
    }
    probs
}

#[cfg(feature = "wgpu")]
#[test]
fn multinomial_without_replacement_wgpu_small_rows() {
    let cats = 6;
    // Row 0 has exactly three nonzero categories, so a 3-sample draw is forced.
    let mut probs = vec![0.0f32, 0.5, 0.0, 0.3, 0.2, 0.0];
    probs.extend_from_slice(&[0.1, 0.2, 0.3, 0.1, 0.2, 0.1]);
    probs.extend_from_slice(&[0.0, 0.4, 0.4, 0.0, 0.1, 0.1]);
    probs.extend_from_slice(&[0.25, 0.0, 0.25, 0.25, 0.25, 0.0]);
    let rows = probs.len() / cats;

    with_wgpu_backend_or_skip(|client, device| {
        let tensor = Tensor::<WgpuRuntime>::from_slice(&probs, &[rows, cats], &device)
            .expect("wgpu probs tensor");
        for _ in 0..8 {
            let out = client
                .multinomial(&tensor, 3, false)
                .expect("wgpu multinomial without replacement");
            assert_eq!(out.shape(), &[rows, 3], "output shape");
            let indices = out.to_vec::<i32>();
            check_draw(&indices, &probs, rows, cats, 3);
            let mut first: Vec<i32> = indices[..3].to_vec();
            first.sort_unstable();
            assert_eq!(first, vec![1, 3, 4], "forced row returns its nonzero set");
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn multinomial_without_replacement_wgpu_many_workgroups() {
    let (rows, cats, samples) = (300, 8, 6);
    let probs = striped_probs(rows, cats);

    with_wgpu_backend_or_skip(|client, device| {
        let tensor = Tensor::<WgpuRuntime>::from_slice(&probs, &[rows, cats], &device)
            .expect("wgpu probs tensor");
        let out = client
            .multinomial(&tensor, samples, false)
            .expect("wgpu multinomial without replacement");
        assert_eq!(out.shape(), &[rows, samples], "output shape");
        check_draw(&out.to_vec::<i32>(), &probs, rows, cats, samples);
    });
}
