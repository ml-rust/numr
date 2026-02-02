//! Statistical operations for WebGPU runtime

use crate::error::Result;
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, StatisticalOps, UnaryOps};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::tensor::Tensor;

impl StatisticalOps<WgpuRuntime> for WgpuClient {
    fn var(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let shape = a.shape();
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| shape[d]).product()
        };

        let mean_val = self.mean(a, dims, true)?;
        let diff = self.sub(a, &mean_val)?;
        let diff_squared = self.square(&diff)?;
        let sum_sq = self.sum(&diff_squared, dims, keepdim)?;
        let divisor = (count.saturating_sub(correction)).max(1) as f64;
        self.div_scalar(&sum_sq, divisor)
    }

    fn std(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let variance = self.var(a, dims, keepdim, correction)?;
        self.sqrt(&variance)
    }

    fn quantile(
        &self,
        a: &Tensor<WgpuRuntime>,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> Result<Tensor<WgpuRuntime>> {
        crate::runtime::wgpu::statistics::quantile_impl(self, a, q, dim, keepdim, interpolation)
    }

    fn percentile(
        &self,
        a: &Tensor<WgpuRuntime>,
        p: f64,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        crate::runtime::wgpu::statistics::percentile_impl(self, a, p, dim, keepdim)
    }

    fn median(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        crate::runtime::wgpu::statistics::median_impl(self, a, dim, keepdim)
    }

    fn histogram(
        &self,
        a: &Tensor<WgpuRuntime>,
        bins: usize,
        range: Option<(f64, f64)>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        crate::runtime::wgpu::statistics::histogram_impl(self, a, bins, range)
    }

    fn cov(&self, a: &Tensor<WgpuRuntime>, ddof: Option<usize>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<WgpuRuntime>>::cov(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<WgpuRuntime>>::corrcoef(self, a)
    }

    fn skew(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        crate::runtime::wgpu::statistics::skew_impl(self, a, dims, keepdim, correction)
    }

    fn kurtosis(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        crate::runtime::wgpu::statistics::kurtosis_impl(self, a, dims, keepdim, correction)
    }

    fn mode(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        crate::runtime::wgpu::statistics::mode_impl(self, a, dim, keepdim)
    }
}
