//! `Kernels::with_level` only returns levels this CPU supports.

use super::common::ALL_LEVELS;
use numr::distance::{Kernels, SimdLevel};

#[test]
fn with_level_only_returns_supported_levels() {
    let detected = Kernels::detect().level();
    assert_eq!(Kernels::with_level(detected), Some(Kernels::detect()));
    assert_eq!(Kernels::scalar().level(), SimdLevel::Scalar);
    assert_eq!(
        Kernels::with_level(SimdLevel::Scalar),
        Some(Kernels::scalar())
    );
    assert_eq!(
        Kernels::with_level(SimdLevel::Avx512).is_some(),
        detected == SimdLevel::Avx512
    );
    for level in ALL_LEVELS {
        if let Some(k) = Kernels::with_level(level) {
            assert_eq!(k.level(), level);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            assert_eq!(Kernels::with_level(SimdLevel::Avx512), None);
        }
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            assert_eq!(Kernels::with_level(SimdLevel::Avx2Fma), None);
        }
        assert_eq!(Kernels::with_level(SimdLevel::Neon), None);
        assert_eq!(Kernels::with_level(SimdLevel::NeonFp16), None);
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(Kernels::with_level(SimdLevel::Avx512), None);
        assert_eq!(Kernels::with_level(SimdLevel::Avx2Fma), None);
        assert!(Kernels::with_level(SimdLevel::Neon).is_some());
    }
}
