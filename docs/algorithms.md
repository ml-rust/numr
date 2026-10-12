# Algorithms

This page lists the algorithm modules in numr. Each module builds on the tensor operations in
[Operations](operations.md).

## Linear Algebra (`numr::algorithm::linalg`)

- **Decompositions**: LU (`lu_decompose`), QR (`qr_decompose`), Cholesky (`cholesky_decompose`),
  SVD (`svd_decompose`), Schur (`schur_decompose`), full eigendecomposition (`eig_decompose`),
  and generalized eigenvalues through the generalized Schur (QZ) decomposition (`qz_decompose`).
- **Solvers**: `solve`, `lstsq`, `pinverse`
- **Matrix functions**: exp (`expm`), log (`logm`), sqrt (`sqrtm`), sign (`signm`)
- **Utilities**: `det`, `trace`, `matrix_rank`, matrix norms

## Fast Fourier Transform (`numr::fft`)

- FFT and IFFT in 1D, 2D, and ND, using the Stockham algorithm
- Real FFT: RFFT and IRFFT

## Matrix Multiplication (`numr::matmul`)

- Tiled GEMM with register blocking
- Bias fusion

## Special Functions (`numr::special`)

- Error functions: `erf`, `erfc`, `erfinv`
- Gamma functions: `gamma`, `lgamma`, `digamma`
- Beta functions: `beta`, `betainc`
- Incomplete gamma: `gammainc`, `gammaincc`
- Bessel functions: J₀, J₁, Y₀, Y₁, I₀, I₁, K₀, K₁
- Elliptic integrals: `ellipk`, `ellipe`
- Hypergeometric functions: `hyp2f1`, `hyp1f1`
- Airy functions: `airy_ai`, `airy_bi`
- Legendre functions: `legendre_p`, `legendre_p_assoc`, `sph_harm`
- Fresnel integrals: `fresnel_s`, `fresnel_c`

## Polynomial Operations (`numr::polynomial`)

- `polyroots`, `polyval`, `polyfromroots`, `polymul`

## Iterative Solvers (`numr::iterative`)

- **Linear solvers**: CG, MINRES, BiCGSTAB, GMRES, LGMRES, CGS, QMR, Jacobi, SOR, Adaptive GMRES
- **Eigensolvers**: Lanczos (symmetric matrices), Arnoldi with implicit restart (IRAM, non-symmetric matrices)
- **Sparse SVD**: `svds`
- **Preconditioners**: ILU(0), IC(0), and algebraic multigrid (AMG) with V-cycles

## Sparse Tensors (`numr::sparse`, feature `sparse`)

- Formats: CSR, CSC, COO
- Operations: SpGEMM (sparse matrix multiplication), SpMV (sparse matrix-vector), DSMM (dense-sparse matrix)
- 2:4 structured sparsity, on the CPU, CUDA, and WebGPU backends

## Sparse Linear Algebra (`numr::sparse_linalg`)

- **Direct solvers**: sparse LU (Gilbert-Peierls) and sparse QR
- **Incomplete factorizations**: ILU(0), ILU(k), IC(0)
- **Preprocessing**: COLAMD ordering and maximum transversal
- **Symbolic and numeric split**: Reuse the sparsity structure for repeated solves

## Related Pages

- [Operations](operations.md): the tensor operations these modules use.
- [Backends](backends.md): dtypes and backend support per operation.
