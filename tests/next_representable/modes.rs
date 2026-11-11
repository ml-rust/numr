//! Floating control registers are thread-local. Run mode checks on a dedicated thread.

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod native {
    use super::super::fixtures::*;
    use numr::prelude::*;

    struct Restore(u64);

    impl Drop for Restore {
        fn drop(&mut self) {
            write_control(self.0);
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn read_control() -> u64 {
        let mut value = 0u32;
        // SAFETY: MXCSR stores four bytes into a valid local variable.
        unsafe {
            std::arch::asm!("stmxcsr [{}]", in(reg) &mut value, options(nostack, preserves_flags));
        }
        u64::from(value)
    }

    #[cfg(target_arch = "x86_64")]
    fn write_control(value: u64) {
        let value = value as u32;
        // SAFETY: Values retain reserved bits from the original MXCSR register.
        unsafe {
            std::arch::asm!("ldmxcsr [{}]", in(reg) &value, options(nostack, preserves_flags));
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn read_control() -> u64 {
        let value: u64;
        // SAFETY: Reading FPCR does not modify memory or control flow.
        unsafe {
            std::arch::asm!("mrs {}, fpcr", out(reg) value, options(nomem, nostack, preserves_flags));
        }
        value
    }

    #[cfg(target_arch = "aarch64")]
    fn write_control(value: u64) {
        // SAFETY: Values retain reserved bits from the original FPCR register.
        unsafe {
            std::arch::asm!("msr fpcr, {}", in(reg) value, options(nomem, nostack, preserves_flags));
        }
    }

    #[test]
    fn neighbors_ignore_rounding_and_subnormal_modes() {
        std::thread::spawn(|| -> Result<()> {
            let device = CpuDevice::new();
            let client = CpuRuntime::default_client(&device);
            let bits32 = &f32_bits()[..30];
            let bits64 = &f64_bits()[..30];
            let values32: Vec<_> = bits32.iter().copied().map(f32::from_bits).collect();
            let values64: Vec<_> = bits64.iter().copied().map(f64::from_bits).collect();
            let input32 = Tensor::from_slice(&values32, &[values32.len()], &device)?;
            let input64 = Tensor::from_slice(&values64, &[values64.len()], &device)?;
            let expected32: Vec<Vec<_>> = [true, false]
                .into_iter()
                .map(|up| bits32.iter().map(|&b| reference_f32(b, up)).collect())
                .collect();
            let expected64: Vec<Vec<_>> = [true, false]
                .into_iter()
                .map(|up| bits64.iter().map(|&b| reference_f64(b, up)).collect())
                .collect();
            let original = read_control();
            let restore = Restore(original);
            for rounding in 0..4u64 {
                #[cfg(target_arch = "x86_64")]
                let controls: Vec<_> = (0..4u64)
                    .map(|mode| {
                        (original & !((3 << 13) | (1 << 15) | (1 << 6)))
                            | (rounding << 13)
                            | ((mode & 1) << 15)
                            | ((mode >> 1) << 6)
                    })
                    .collect();
                #[cfg(target_arch = "aarch64")]
                let controls: Vec<_> = (0..2u64)
                    .map(|flush| {
                        (original & !((3 << 22) | (1 << 24))) | (rounding << 22) | (flush << 24)
                    })
                    .collect();
                for control in controls {
                    write_control(control);
                    for (direction, up) in [true, false].into_iter().enumerate() {
                        let out32 = if up {
                            client.next_up(&input32)?
                        } else {
                            client.next_down(&input32)?
                        };
                        let out64 = if up {
                            client.next_up(&input64)?
                        } else {
                            client.next_down(&input64)?
                        };
                        assert_eq!(
                            out32
                                .try_to_vec::<f32>()?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>(),
                            expected32[direction]
                        );
                        assert_eq!(
                            out64
                                .try_to_vec::<f64>()?
                                .into_iter()
                                .map(f64::to_bits)
                                .collect::<Vec<_>>(),
                            expected64[direction]
                        );
                    }
                }
            }
            drop(restore);
            assert_eq!(read_control(), original);
            Ok(())
        })
        .join()
        .expect("mode thread panicked")
        .expect("mode operation returned an error");
    }
}
