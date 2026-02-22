//! Core Householder QR algorithm for sparse matrices
//!
//! Column-wise left-looking Householder QR with rank detection.

use crate::algorithm::sparse_linalg::qr::types::QrOptions;
use crate::error::{Error, Result};

/// Internal result from numeric QR factorization
pub(crate) struct QrNumericResult {
    pub householder_vectors: Vec<(Vec<i64>, Vec<f64>)>,
    pub tau: Vec<f64>,
    pub r_col_ptrs: Vec<i64>,
    pub r_row_indices: Vec<i64>,
    pub r_values: Vec<f64>,
    pub rank: usize,
}

/// Column-wise left-looking Householder QR factorization
///
/// Processes one column at a time:
/// 1. Apply COLAMD permutation to get A*P
/// 2. For each column k:
///    a. Scatter A*P column k into dense work vector
///    b. Apply previous Householder reflectors to the column
///    c. Compute new Householder reflector from column below diagonal
///    d. Store R entries (above diagonal) and reflector
/// 3. Detect rank from R diagonal
pub(crate) fn householder_qr(
    m: usize,
    n: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
    values: &[f64],
    col_perm: &[usize],
    options: &QrOptions,
) -> Result<QrNumericResult> {
    let min_mn = m.min(n);

    let mut householder_vectors: Vec<(Vec<i64>, Vec<f64>)> = Vec::with_capacity(min_mn);
    let mut tau_vec: Vec<f64> = Vec::with_capacity(min_mn);

    // R stored column by column (dynamically built)
    let mut r_col_ptrs: Vec<i64> = vec![0i64; n + 1];
    let mut r_row_indices: Vec<i64> = Vec::new();
    let mut r_values: Vec<f64> = Vec::new();

    let mut rank = min_mn;

    // Dense work vector for current column
    let mut work = vec![0.0f64; m];

    for k in 0..min_mn {
        // Step 1: Scatter permuted column k into work vector
        let orig_col = col_perm[k];
        let start = col_ptrs[orig_col] as usize;
        let end = col_ptrs[orig_col + 1] as usize;

        work.fill(0.0);
        for idx in start..end {
            let row = row_indices[idx] as usize;
            work[row] = values[idx];
        }

        // Step 2: Apply previous Householder reflectors Q_0..Q_{k-1} to this column
        apply_reflectors(&householder_vectors, &tau_vec, &mut work, k);

        // Step 3: Extract R entries for column k (rows 0..k)
        for row in 0..k {
            if work[row].abs() > 1e-15 {
                r_row_indices.push(row as i64);
                r_values.push(work[row]);
            }
        }

        // Step 4: Compute Householder reflector for work[k..m]
        let (v_indices, v_values, tau, diag_val) = compute_householder(&work, k, m);

        // Store R diagonal entry
        r_row_indices.push(k as i64);
        r_values.push(diag_val);

        r_col_ptrs[k + 1] = r_row_indices.len() as i64;

        // Check rank
        if diag_val.abs() < options.rank_tolerance {
            rank = k;
            householder_vectors.push((v_indices, v_values));
            tau_vec.push(tau);

            process_remaining_columns(
                k + 1,
                min_mn,
                n,
                col_ptrs,
                row_indices,
                values,
                col_perm,
                &mut householder_vectors,
                &mut tau_vec,
                &mut work,
                &mut r_col_ptrs,
                &mut r_row_indices,
                &mut r_values,
            );

            return Ok(QrNumericResult {
                householder_vectors,
                tau: tau_vec,
                r_col_ptrs,
                r_row_indices,
                r_values,
                rank,
            });
        }

        // Store reflector
        householder_vectors.push((v_indices, v_values));
        tau_vec.push(tau);
    }

    // Fill remaining R col_ptrs for columns beyond min_mn (if n > m, they're empty)
    for kk in min_mn..n {
        r_col_ptrs[kk + 1] = r_col_ptrs[min_mn];
    }

    Ok(QrNumericResult {
        householder_vectors,
        tau: tau_vec,
        r_col_ptrs,
        r_row_indices,
        r_values,
        rank,
    })
}

/// Apply Householder reflectors 0..count to a work vector
fn apply_reflectors(
    householder_vectors: &[(Vec<i64>, Vec<f64>)],
    tau_vec: &[f64],
    work: &mut [f64],
    count: usize,
) {
    for j in 0..count {
        let (ref v_indices, ref v_values) = householder_vectors[j];
        let tau_j = tau_vec[j];

        let mut dot = 0.0;
        for (idx, &vi) in v_indices.iter().zip(v_values.iter()) {
            dot += vi * work[*idx as usize];
        }

        let scale = tau_j * dot;
        for (idx, &vi) in v_indices.iter().zip(v_values.iter()) {
            work[*idx as usize] -= scale * vi;
        }
    }
}

/// Process remaining columns after rank deficiency is detected
#[allow(clippy::too_many_arguments)]
fn process_remaining_columns(
    start_col: usize,
    min_mn: usize,
    n: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
    values: &[f64],
    col_perm: &[usize],
    householder_vectors: &mut Vec<(Vec<i64>, Vec<f64>)>,
    tau_vec: &mut Vec<f64>,
    work: &mut [f64],
    r_col_ptrs: &mut [i64],
    r_row_indices: &mut Vec<i64>,
    r_values: &mut Vec<f64>,
) {
    let m = work.len();

    for kk in start_col..min_mn {
        let orig_col2 = col_perm[kk];
        let start2 = col_ptrs[orig_col2] as usize;
        let end2 = col_ptrs[orig_col2 + 1] as usize;

        work.fill(0.0);
        for idx in start2..end2 {
            let row = row_indices[idx] as usize;
            work[row] = values[idx];
        }

        // Apply all previous reflectors (including newly added ones)
        apply_reflectors(
            householder_vectors,
            tau_vec,
            work,
            householder_vectors.len(),
        );

        // Store R column
        for row in 0..=kk {
            if work[row].abs() > 1e-15 || row == kk {
                r_row_indices.push(row as i64);
                r_values.push(work[row]);
            }
        }
        r_col_ptrs[kk + 1] = r_row_indices.len() as i64;

        // Compute and store reflector for this column
        let (vi, vv, t, _dv) = compute_householder(work, kk, m);
        householder_vectors.push((vi, vv));
        tau_vec.push(t);
    }

    // Fill remaining R col_ptrs
    for kk in min_mn..n {
        r_col_ptrs[kk + 1] = r_col_ptrs[kk];
    }
}

/// Compute Householder reflector for x = work[start..m]
///
/// Returns: (v_row_indices, v_values, tau, diagonal_value)
///
/// The reflector satisfies: (I - tau * v * v^T) * x = ||x|| * e_1
pub(crate) fn compute_householder(
    work: &[f64],
    start: usize,
    m: usize,
) -> (Vec<i64>, Vec<f64>, f64, f64) {
    // Compute norm of x = work[start..m]
    let mut norm_sq = 0.0;
    for i in start..m {
        norm_sq += work[i] * work[i];
    }
    let norm = norm_sq.sqrt();

    if norm < 1e-30 {
        // Zero column — no reflector needed
        return (vec![start as i64], vec![1.0], 0.0, 0.0);
    }

    // Choose sign to avoid cancellation: sigma = -sign(x[start]) * ||x||
    let sigma = if work[start] >= 0.0 { -norm } else { norm };
    let diag_val = sigma; // R[start, start] = sigma

    let v_start = work[start] - sigma;

    // Normalize v so that v[start] = 1
    if v_start.abs() < 1e-30 {
        return (vec![start as i64], vec![1.0], 0.0, diag_val);
    }

    let inv_v_start = 1.0 / v_start;

    let mut v_indices = Vec::new();
    let mut v_values = Vec::new();

    v_indices.push(start as i64);
    v_values.push(1.0); // v[start] = 1 (normalized)

    for i in (start + 1)..m {
        if work[i].abs() > 1e-15 {
            v_indices.push(i as i64);
            v_values.push(work[i] * inv_v_start);
        }
    }

    // tau = (sigma - x[start]) / sigma = -v_start / sigma
    let tau = -v_start / sigma;

    (v_indices, v_values, tau, diag_val)
}

/// Apply Q^T to a vector by applying Householder reflectors in forward order.
///
/// Q^T * b is computed as: for j = 0, 1, ..., k-1: b = (I - tau_j * v_j * v_j^T) * b
pub(crate) fn apply_qt(householder_vectors: &[(Vec<i64>, Vec<f64>)], tau: &[f64], b: &mut [f64]) {
    for j in 0..householder_vectors.len() {
        let (ref v_indices, ref v_values) = householder_vectors[j];
        let tau_j = tau[j];

        if tau_j == 0.0 {
            continue;
        }

        let mut dot = 0.0;
        for (idx, &vi) in v_indices.iter().zip(v_values.iter()) {
            dot += vi * b[*idx as usize];
        }

        let scale = tau_j * dot;
        for (idx, &vi) in v_indices.iter().zip(v_values.iter()) {
            b[*idx as usize] -= scale * vi;
        }
    }
}

/// Back-substitute: solve R[0:n, 0:n] * x = rhs
/// R is in CSC format.
pub(crate) fn back_substitute(
    n: usize,
    r_col_ptrs: &[i64],
    r_row_indices: &[i64],
    r_values: &[f64],
    rhs: &[f64],
    x: &mut [f64],
) -> Result<()> {
    x[..n].copy_from_slice(rhs);

    for col in (0..n).rev() {
        let start = r_col_ptrs[col] as usize;
        let end = r_col_ptrs[col + 1] as usize;

        // Find diagonal entry
        let mut diag_val = 0.0;
        for idx in start..end {
            if r_row_indices[idx] as usize == col {
                diag_val = r_values[idx];
                break;
            }
        }

        if diag_val.abs() < 1e-30 {
            return Err(Error::Internal(format!(
                "sparse_qr back_substitute: zero diagonal at column {}",
                col
            )));
        }

        x[col] /= diag_val;

        // Update rows above
        for idx in start..end {
            let row = r_row_indices[idx] as usize;
            if row < col {
                x[row] -= r_values[idx] * x[col];
            }
        }
    }

    Ok(())
}
