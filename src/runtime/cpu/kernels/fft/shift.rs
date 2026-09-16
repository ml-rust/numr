//! FFT shift kernels

use crate::dtype::{Complex64, Complex128};

/// Shift zero-frequency component to center
///
/// For 1D array of length N, swaps [0..N/2] with [N/2..N]
#[allow(clippy::manual_memcpy)]
pub unsafe fn fftshift_c64(input: &[Complex64], output: &mut [Complex64]) {
    let n = input.len();
    let half_n = n / 2;

    // Copy second half to first half of output
    for i in 0..half_n {
        output[i] = input[half_n + i];
    }
    // Copy first half to second half of output
    for i in 0..n - half_n {
        output[half_n + i] = input[i];
    }
}

/// Inverse shift (undo fftshift)
#[allow(clippy::manual_memcpy, clippy::manual_div_ceil)]
pub unsafe fn ifftshift_c64(input: &[Complex64], output: &mut [Complex64]) {
    let n = input.len();
    let half_n = (n + 1) / 2; // For odd lengths, first half is larger

    // For ifftshift: swap [0..ceil(N/2)] with [ceil(N/2)..N]
    let shift = n - half_n;
    for i in 0..shift {
        output[i] = input[half_n + i];
    }
    for i in 0..half_n {
        output[shift + i] = input[i];
    }
}

/// Shift zero-frequency component to center (f64)
#[allow(clippy::manual_memcpy)]
pub unsafe fn fftshift_c128(input: &[Complex128], output: &mut [Complex128]) {
    let n = input.len();
    let half_n = n / 2;

    for i in 0..half_n {
        output[i] = input[half_n + i];
    }
    for i in 0..n - half_n {
        output[half_n + i] = input[i];
    }
}

/// Inverse shift (f64)
#[allow(clippy::manual_memcpy, clippy::manual_div_ceil)]
pub unsafe fn ifftshift_c128(input: &[Complex128], output: &mut [Complex128]) {
    let n = input.len();
    let half_n = (n + 1) / 2;

    let shift = n - half_n;
    for i in 0..shift {
        output[i] = input[half_n + i];
    }
    for i in 0..half_n {
        output[shift + i] = input[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fftshift() {
        let input = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let mut output = [Complex64::default(); 4];

        unsafe {
            fftshift_c64(&input, &mut output);
        }

        // [0, 1, 2, 3] -> [2, 3, 0, 1]
        assert!((output[0].re - 2.0).abs() < 1e-5);
        assert!((output[1].re - 3.0).abs() < 1e-5);
        assert!((output[2].re - 0.0).abs() < 1e-5);
        assert!((output[3].re - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fftshift_ifftshift_roundtrip() {
        let original = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ];
        let mut shifted = [Complex64::default(); 4];
        let mut unshifted = [Complex64::default(); 4];

        unsafe {
            fftshift_c64(&original, &mut shifted);
            ifftshift_c64(&shifted, &mut unshifted);
        }

        for i in 0..4 {
            assert!((unshifted[i].re - original[i].re).abs() < 1e-5);
            assert!((unshifted[i].im - original[i].im).abs() < 1e-5);
        }
    }
}
