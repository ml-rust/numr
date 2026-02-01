//! Statistical operations for CUDA runtime
use crate::error::Result;
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, StatisticalOps, UnaryOps};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

// Import helper functions from statistics module
use crate::runtime::cuda::ops::statistics::{
    histogram_impl, kurtosis_impl, median_impl, mode_impl, percentile_impl, quantile_impl,
    skew_impl,
};

impl StatisticalOps<CudaRuntime> for CudaClient {
    fn var(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Variance implementation using existing ops
        // var(x) = mean((x - mean(x))^2) * N / (N - correction)

        let shape = a.shape();

        // When dims is empty, reduce over all dimensions
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..shape.len()).collect()
        } else {
            dims.to_vec()
        };

        // Compute count of elements being reduced
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| shape[d]).product()
        };

        // Compute mean (mean already handles empty dims internally)
        let mean_val = self.mean(a, dims, true)?;

        // Compute (x - mean)
        let diff = self.sub(a, &mean_val)?;

        // Compute (x - mean)^2
        let diff_squared = self.square(&diff)?;

        // Compute sum of squared differences over all dims when dims is empty
        let sum_sq = self.sum(&diff_squared, &actual_dims, keepdim)?;

        // Divide by (N - correction)
        let divisor = (count.saturating_sub(correction)).max(1) as f64;
        self.div_scalar(&sum_sq, divisor)
    }

    fn std(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Standard deviation is sqrt of variance
        let variance = self.var(a, dims, keepdim, correction)?;
        self.sqrt(&variance)
    }

    fn quantile(
        &self,
        a: &Tensor<CudaRuntime>,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> Result<Tensor<CudaRuntime>> {
        quantile_impl(self, a, q, dim, keepdim, interpolation)
    }

    fn percentile(
        &self,
        a: &Tensor<CudaRuntime>,
        p: f64,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        percentile_impl(self, a, p, dim, keepdim)
    }

    fn median(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        median_impl(self, a, dim, keepdim)
    }

    fn histogram(
        &self,
        a: &Tensor<CudaRuntime>,
        bins: usize,
        range: Option<(f64, f64)>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        histogram_impl(self, a, bins, range)
    }

    fn cov(&self, a: &Tensor<CudaRuntime>, ddof: Option<usize>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<CudaRuntime>>::cov(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<CudaRuntime>>::corrcoef(self, a)
    }

    fn skew(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        skew_impl(self, a, dims, keepdim, correction)
    }

    fn kurtosis(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        kurtosis_impl(self, a, dims, keepdim, correction)
    }

    fn mode(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        mode_impl(self, a, dim, keepdim)
    }
}
