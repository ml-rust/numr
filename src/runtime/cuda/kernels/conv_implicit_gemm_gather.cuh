// Gather body of the implicit-GEMM conv2d in conv_implicit_gemm.cuh.
//
// Row m of a block is GEMM row blockIdx.x * BM + m. K runs tap-major and
// every K-tile gathers its A rows from global memory. The pipeline is
// described at the top of conv_implicit_gemm.cuh.

#ifndef NUMR_CONV_IMPLICIT_GEMM_GATHER_CUH
#define NUMR_CONV_IMPLICIT_GEMM_GATHER_CUH

/* Gather body. See the pipeline at the top of conv_implicit_gemm.cuh. */
template <typename T, typename Cfg, bool VEC, bool HALF_ACC>
__device__ __forceinline__ void conv_igemm_run(const ConvIgemmParams<T>& p) {
    conv_check_acc<T, Cfg, HALF_ACC>();
    using Loader = typename ConvLoaderFor<T, Cfg, VEC>::type;
    using Acc = ConvAccum<HALF_ACC>;

    __shared__ __align__(128) unsigned char smem_pool[Cfg::POOL_BYTES];
    __shared__ __align__(16) longlong2 smem_rows[Cfg::BM];
    __shared__ __align__(16) int4 smem_cols[VEC ? 1 : 2 * Cfg::BK];

    /* Read once through volatile asm, so the compiler keeps the values
       derived from it in registers instead of recomputing them from the
       special register inside the K loop. */
    int tid;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / Cfg::WARPS_N;
    const int warp_n = warp % Cfg::WARPS_N;
    const long long block_row = static_cast<long long>(blockIdx.x) * Cfg::BM;
    const int block_col = static_cast<int>(blockIdx.y) * Cfg::BN;

    conv_fill_row_table<T, Cfg>(p, smem_rows, tid, block_row);

    Loader ld;
    ld.init(p, tid, block_row, block_col);
    const int num_k_tiles = Loader::num_k_tiles(p);
    const ConvLdmOffsets<Cfg> ldm(lane, warp_m, warp_n);
    const unsigned int pool = conv_smem_addr(smem_pool);
    const unsigned int sts = pool + ld.sts_off;
    /* Stage the current K-tile reads from and the next one lands in. Both
       toggle by XOR, so no stage index math runs in the loop. */
    unsigned int rd = 0u;
    unsigned int wr = Cfg::STAGE_BYTES;

    typename Acc::type acc[Cfg::FRAG_M][Cfg::FRAG_N][Acc::REGS];
    conv_zero_acc<Cfg, HALF_ACC>(acc);

    /* Column tables of K-tiles 0 and 1. Table t % 2 serves K-tile t. */
    if (!VEC) {
        ld.write_cols(p, smem_cols, tid);
        ld.write_cols(p, smem_cols + Cfg::BK, tid);
        __syncthreads();
    }
    ConvStageRegs<Cfg> regs;
    ld.load(p, smem_cols, regs);
    conv_stage_store<Cfg>(sts, regs);
    if (num_k_tiles > 1) {
        ld.advance(p);
        ld.load(p, smem_cols + Cfg::BK, regs);
    }
    __syncthreads();

    ConvFrags<Cfg> f0;
    ConvFrags<Cfg> f1;
    if constexpr (VEC) {
        conv_load_frags<Cfg, 0>(f0, pool, ldm);
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            const bool more = kt + 1 < num_k_tiles;
            conv_load_frags<Cfg, 1>(f1, pool + rd, ldm);
            conv_mma_k16<T, Cfg, HALF_ACC>(acc, f0);
            /* The other stage was last read before the previous barrier. */
            if (more) conv_stage_store<Cfg>(sts + wr, regs);
            /* Publishes the stage written above, and orders every ldmatrix
               of stage `rd` before its next overwrite. After the last
               K-tile it frees the stages for the epilogue's output tile. */
            __syncthreads();
            if (more) conv_load_frags<Cfg, 0>(f0, pool + wr, ldm);
            if (kt + 2 < num_k_tiles) {
                ld.advance(p);
                ld.load(p, smem_cols, regs);
            }
            conv_mma_k16<T, Cfg, HALF_ACC>(acc, f1);
            rd ^= Cfg::STAGE_BYTES;
            wr ^= Cfg::STAGE_BYTES;
        }
    } else {
        /* The scalar gather packs 16-bit loads into chunks and keeps many
           temporaries in flight. Its loads issue at the top of the K-tile,
           before any fragment is live, and its fragments load inside the
           K-tile. */
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            const bool more = kt + 1 < num_k_tiles;
            const int table = (kt & 1) * Cfg::BK;
            if (kt >= 1 && more) {
                ld.advance(p);
                ld.load(p, smem_cols + (Cfg::BK - table), regs);
            }
            conv_load_frags<Cfg, 0>(f0, pool + rd, ldm);
            conv_load_frags<Cfg, 1>(f1, pool + rd, ldm);
            conv_mma_k16<T, Cfg, HALF_ACC>(acc, f0);
            conv_mma_k16<T, Cfg, HALF_ACC>(acc, f1);
            if (more) conv_stage_store<Cfg>(sts + wr, regs);
            /* Table kt % 2 last served the load of K-tile kt, before the
               previous barrier. It now takes K-tile kt + 2. */
            ld.write_cols(p, smem_cols + table, tid);
            /* Publishes the stage and table written above, and orders every
               ldmatrix of stage `rd` before its next overwrite. */
            __syncthreads();
            rd ^= Cfg::STAGE_BYTES;
            wr ^= Cfg::STAGE_BYTES;
        }
    }

    conv_finish<T, Cfg, HALF_ACC>(p, acc, smem_pool, smem_rows, tid, warp_m, warp_n, block_col);
}

#endif  // NUMR_CONV_IMPLICIT_GEMM_GATHER_CUH
