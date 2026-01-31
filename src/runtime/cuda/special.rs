//! CUDA implementation of special mathematical functions
//!
//! This module implements the [`SpecialFunctions`] trait for CUDA using native
//! CUDA math library functions. CUDA provides built-in functions for erf, erfc,
//! lgamma, tgamma (gamma), etc.
//!
//! Native CUDA kernels are used - NO cuBLAS/cuSPARSE dependency.

use super::CudaRuntime;
use super::client::CudaClient;
use super::kernels;
use crate::algorithm::special::{SpecialFunctions, validate_special_dtype};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

impl SpecialFunctions<CudaRuntime> for CudaClient {
    fn erf(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_erf(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn erfc(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_erfc(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn erfinv(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_erfinv(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn gamma(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_gamma(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn lgamma(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_lgamma(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn digamma(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_digamma(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn beta(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(a.shape(), a.dtype(), device);

        unsafe {
            kernels::launch_beta(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.storage().ptr(),
                b.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn betainc(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() || a.dtype() != x.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() || a.shape() != x.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(a.shape(), a.dtype(), device);

        unsafe {
            kernels::launch_betainc(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.storage().ptr(),
                b.storage().ptr(),
                x.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn gammainc(
        &self,
        a: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != x.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: x.dtype(),
            });
        }
        if a.shape() != x.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(a.shape(), a.dtype(), device);

        unsafe {
            kernels::launch_gammainc(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.storage().ptr(),
                x.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn gammaincc(
        &self,
        a: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != x.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: x.dtype(),
            });
        }
        if a.shape() != x.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(a.shape(), a.dtype(), device);

        unsafe {
            kernels::launch_gammaincc(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.storage().ptr(),
                x.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn gammaincinv(
        &self,
        a: &Tensor<CudaRuntime>,
        p: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != p.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: p.dtype(),
            });
        }
        if a.shape() != p.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: p.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(a.shape(), a.dtype(), device);

        unsafe {
            kernels::launch_gammaincinv(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.storage().ptr(),
                p.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn betaincinv(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        p: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() || a.dtype() != p.dtype() {
            return Err(crate::error::Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() || a.shape() != p.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: p.shape().to_vec(),
            });
        }

        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(a.shape(), a.dtype(), device);

        unsafe {
            kernels::launch_betaincinv(
                self.context(),
                self.stream(),
                device.index,
                a.dtype(),
                a.storage().ptr(),
                b.storage().ptr(),
                p.storage().ptr(),
                out.storage().ptr(),
                a.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_j0(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_bessel_j0(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_j1(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_bessel_j1(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_y0(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_bessel_y0(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_y1(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_bessel_y1(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_i0(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_bessel_i0(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_i1(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_bessel_i1(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_k0(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_bessel_k0(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }

    fn bessel_k1(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_special_dtype(x.dtype())?;
        let device = self.device();
        let out = Tensor::<CudaRuntime>::empty(x.shape(), x.dtype(), device);

        unsafe {
            kernels::launch_bessel_k1(
                self.context(),
                self.stream(),
                device.index,
                x.dtype(),
                x.storage().ptr(),
                out.storage().ptr(),
                x.numel(),
            )?;
        }

        Ok(out)
    }
}
