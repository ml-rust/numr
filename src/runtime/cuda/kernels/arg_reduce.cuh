// Merge rule for argmax and argmin, shared by `reduce.cu` and `reduce_int.cu`.
//
// The rule matches `runtime/cpu/kernels/reduce/special.rs`:
// - Equal values resolve to the lowest index.
// - NaN never wins. A row of only NaN returns index 0.
//
// A thread scans indices tid, tid + blockDim, ... so a strict comparison alone
// keeps the lowest thread, not the lowest index. Each merge must compare the
// index on a tie.

#pragma once

// Index of a lane that holds no candidate yet. Every real index is lower.
#define ARG_NO_INDEX 0x7fffffffffffffffLL

// True when candidate (v, i) replaces the current best (cur, cur_i).
// NaN fails both `>`/`<` and `==`, so a NaN candidate never replaces anything.
template<bool IsMax, typename V>
__device__ __forceinline__ bool arg_take(V v, long long i, V cur, long long cur_i) {
    return (IsMax ? (v > cur) : (v < cur)) || (v == cur && i < cur_i);
}
