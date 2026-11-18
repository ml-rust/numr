// Device functions of CorrectlyRoundedOps: cr_exp, cr_log, cr_log1p and cr_pow.
//
// Each function mirrors the Rust lane kernel in src/algorithm/correctly_rounded/ line for
// line, so CUDA and CPU return the same bits. The rules every part follows:
// - Every double operation is __dadd_rn, __dsub_rn, __dmul_rn or __fma_rn, in the Rust
//   order. No libdevice math function appears.
// - The file compiles with --fmad=false and without --use_fast_math.
// - Integer work on bit patterns is exact; 128-bit products use __umul64hi.
// - The tables come from correctly_rounded_tables.cuh, which build.rs generates from
//   src/algorithm/correctly_rounded/tables_data.rs.

#ifndef NUMR_CORRECTLY_ROUNDED_LANE_CUH
#define NUMR_CORRECTLY_ROUNDED_LANE_CUH

#include "correctly_rounded_exp.cuh"
#include "correctly_rounded_log.cuh"
#include "correctly_rounded_log1p.cuh"
#include "correctly_rounded_pow.cuh"

#endif
