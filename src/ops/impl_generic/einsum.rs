//! Generic einsum implementation composing primitive operations.
//!
//! Einsum is a composite operation that uses transpose/permute, reshape, matmul,
//! mul, and sum to implement arbitrary Einstein summation expressions.

use crate::error::{Error, Result};
use crate::ops::{BinaryOps, MatmulOps, ReduceOps, ScalarOps};
use crate::runtime::Runtime;
use crate::tensor::{Layout, Tensor};
use std::collections::{BTreeSet, HashMap};

/// Parsed einsum notation.
#[derive(Debug, Clone)]
struct EinsumParsed {
    /// Subscript labels for each input tensor.
    input_subscripts: Vec<Vec<char>>,
    /// Subscript labels for the output tensor.
    output_subscripts: Vec<char>,
}

/// Parse einsum notation string.
///
/// Supports explicit notation ("ij,jk->ik") and implicit notation ("ij,jk").
/// For implicit notation, the output subscripts are the sorted list of labels
/// appearing exactly once across all inputs.
fn parse_notation(notation: &str, num_inputs: usize) -> Result<EinsumParsed> {
    let notation = notation.trim();

    let (inputs_str, output_str) = if let Some(arrow_pos) = notation.find("->") {
        let inputs = &notation[..arrow_pos];
        let output = &notation[arrow_pos + 2..];
        (inputs, Some(output))
    } else {
        (notation, None)
    };

    let input_parts: Vec<&str> = inputs_str.split(',').collect();
    if input_parts.len() != num_inputs {
        return Err(Error::InvalidArgument {
            arg: "notation",
            reason: format!(
                "einsum notation specifies {} inputs but {} were provided",
                input_parts.len(),
                num_inputs
            ),
        });
    }

    let mut input_subscripts = Vec::with_capacity(num_inputs);
    for part in &input_parts {
        let subs: Vec<char> = part.chars().collect();
        for &c in &subs {
            if !c.is_ascii_lowercase() {
                return Err(Error::InvalidArgument {
                    arg: "notation",
                    reason: format!("einsum subscript must be lowercase letter, got '{c}'"),
                });
            }
        }
        input_subscripts.push(subs);
    }

    let output_subscripts = if let Some(out_str) = output_str {
        let subs: Vec<char> = out_str.chars().collect();
        for &c in &subs {
            if !c.is_ascii_lowercase() {
                return Err(Error::InvalidArgument {
                    arg: "notation",
                    reason: format!("einsum output subscript must be lowercase letter, got '{c}'"),
                });
            }
        }
        subs
    } else {
        // Implicit mode: output = sorted labels appearing exactly once
        let mut counts: HashMap<char, usize> = HashMap::new();
        for subs in &input_subscripts {
            for &c in subs {
                *counts.entry(c).or_insert(0) += 1;
            }
        }
        let mut output: Vec<char> = counts
            .into_iter()
            .filter(|&(_, count)| count == 1)
            .map(|(c, _)| c)
            .collect();
        output.sort();
        output
    };

    Ok(EinsumParsed {
        input_subscripts,
        output_subscripts,
    })
}

/// Validate that tensor shapes match the subscripts.
fn validate_shapes<R: Runtime>(
    parsed: &EinsumParsed,
    inputs: &[&Tensor<R>],
) -> Result<HashMap<char, usize>> {
    let mut label_sizes: HashMap<char, usize> = HashMap::new();

    for (i, (subs, tensor)) in parsed
        .input_subscripts
        .iter()
        .zip(inputs.iter())
        .enumerate()
    {
        if subs.len() != tensor.ndim() {
            return Err(Error::InvalidArgument {
                arg: "notation",
                reason: format!(
                    "input {} has {} subscripts but tensor has {} dimensions",
                    i,
                    subs.len(),
                    tensor.ndim()
                ),
            });
        }

        for (dim_idx, &label) in subs.iter().enumerate() {
            let size = tensor.shape()[dim_idx];
            if let Some(&existing) = label_sizes.get(&label) {
                if existing != size {
                    return Err(Error::InvalidArgument {
                        arg: "notation",
                        reason: format!(
                            "dimension mismatch for label '{}': {} vs {}",
                            label, existing, size
                        ),
                    });
                }
            } else {
                label_sizes.insert(label, size);
            }
        }
    }

    // Validate output labels exist in inputs
    for &label in &parsed.output_subscripts {
        if !label_sizes.contains_key(&label) {
            return Err(Error::InvalidArgument {
                arg: "notation",
                reason: format!("output label '{label}' not found in any input"),
            });
        }
    }

    Ok(label_sizes)
}

/// Generic einsum implementation using primitive operations.
///
/// Strategy:
/// 1. For single-input: permute + reshape + sum over contracted dims
/// 2. For two-input: try to map to matmul, otherwise fall back to
///    broadcast-multiply + sum approach
/// 3. For N>2 inputs: left-fold pairwise contractions
pub fn einsum_impl<R, C>(client: &C, notation: &str, inputs: &[&Tensor<R>]) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + ReduceOps<R> + MatmulOps<R> + ScalarOps<R>,
{
    if inputs.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "inputs",
            reason: "einsum requires at least one input".to_string(),
        });
    }

    let parsed = parse_notation(notation, inputs.len())?;
    let label_sizes = validate_shapes(&parsed, inputs)?;

    let result = match inputs.len() {
        1 => einsum_unary(client, &parsed, inputs[0], &label_sizes),
        2 => einsum_binary(client, &parsed, inputs[0], inputs[1], &label_sizes),
        _ => einsum_nary(client, notation, inputs),
    }?;

    // Ensure result is contiguous for downstream consumers
    Ok(result.contiguous())
}

/// Single-input einsum: trace, transpose, diagonal sum, partial sum.
fn einsum_unary<R, C>(
    client: &C,
    parsed: &EinsumParsed,
    input: &Tensor<R>,
    _label_sizes: &HashMap<char, usize>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + ReduceOps<R> + MatmulOps<R> + ScalarOps<R>,
{
    let in_subs = &parsed.input_subscripts[0];
    let out_subs = &parsed.output_subscripts;

    // Check for repeated labels (trace-like operations)
    let mut label_dims: HashMap<char, Vec<usize>> = HashMap::new();
    for (dim, &label) in in_subs.iter().enumerate() {
        label_dims.entry(label).or_default().push(dim);
    }

    let has_repeated = label_dims.values().any(|dims| dims.len() > 1);

    if has_repeated {
        return einsum_unary_with_trace(client, parsed, input);
    }

    // No repeated labels: just permute and sum over non-output dims
    let output_set: BTreeSet<char> = out_subs.iter().copied().collect();

    // Find dims to sum over (in input but not in output)
    let sum_dims: Vec<usize> = in_subs
        .iter()
        .enumerate()
        .filter(|(_, label)| !output_set.contains(label))
        .map(|(dim, _)| dim)
        .collect();

    let mut result = input.clone();

    // Sum over contracted dims (from highest to lowest to keep indices valid)
    if !sum_dims.is_empty() {
        result = client.sum(&result, &sum_dims, false)?;
    }

    // After summing, remaining dims correspond to labels not summed
    // We need to permute to match output order
    let remaining_labels: Vec<char> = in_subs
        .iter()
        .enumerate()
        .filter(|(dim, _)| !sum_dims.contains(dim))
        .map(|(_, &label)| label)
        .collect();

    if remaining_labels != *out_subs && !out_subs.is_empty() {
        let perm: Vec<usize> = out_subs
            .iter()
            .map(|label| {
                remaining_labels
                    .iter()
                    .position(|l| l == label)
                    .unwrap_or(0)
            })
            .collect();

        if perm.len() > 1 {
            result = result.permute(&perm)?;
        }
    }

    Ok(result)
}

/// Extract diagonal of a 2D tensor as a 1D view using stride tricks.
fn extract_diagonal<R: Runtime>(input: &Tensor<R>) -> Result<Tensor<R>> {
    let n = input.shape()[0];
    let layout = input.layout();
    let diag_stride = layout.strides()[0] + layout.strides()[1];
    let diag_layout = Layout::new(
        smallvec::smallvec![n],
        smallvec::smallvec![diag_stride],
        layout.offset(),
    );
    Ok(Tensor::from_parts(input.storage().clone(), diag_layout))
}

/// Handle trace-like unary einsum (repeated labels).
fn einsum_unary_with_trace<R, C>(
    client: &C,
    parsed: &EinsumParsed,
    input: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + ReduceOps<R> + MatmulOps<R> + ScalarOps<R>,
{
    let in_subs = &parsed.input_subscripts[0];
    let out_subs = &parsed.output_subscripts;

    // "ii->" : trace (sum of diagonal)
    if in_subs.len() == 2 && in_subs[0] == in_subs[1] && out_subs.is_empty() {
        let diag = extract_diagonal(input)?;
        return client.sum(&diag, &[0], false);
    }

    // "ii->i" : diagonal extraction
    if in_subs.len() == 2 && in_subs[0] == in_subs[1] && out_subs.len() == 1 {
        let diag = extract_diagonal(input)?;
        return Ok(diag.contiguous());
    }

    Err(Error::NotImplemented {
        feature: "einsum with repeated labels in complex patterns",
    })
}

/// Two-input einsum: try matmul path, otherwise broadcast-multiply + sum.
fn einsum_binary<R, C>(
    client: &C,
    parsed: &EinsumParsed,
    a: &Tensor<R>,
    b: &Tensor<R>,
    label_sizes: &HashMap<char, usize>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + ReduceOps<R> + MatmulOps<R> + ScalarOps<R>,
{
    let a_subs = &parsed.input_subscripts[0];
    let b_subs = &parsed.input_subscripts[1];
    let out_subs = &parsed.output_subscripts;

    if let Some(result) = try_matmul_path(client, a_subs, b_subs, out_subs, a, b)? {
        return Ok(result);
    }

    einsum_binary_general(client, a_subs, b_subs, out_subs, a, b, label_sizes)
}

/// Try to map a binary einsum to matmul.
fn try_matmul_path<R, C>(
    client: &C,
    a_subs: &[char],
    b_subs: &[char],
    out_subs: &[char],
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Option<Tensor<R>>>
where
    R: Runtime,
    C: BinaryOps<R> + ReduceOps<R> + MatmulOps<R> + ScalarOps<R>,
{
    // Simple matmul: "ij,jk->ik"
    if a_subs.len() == 2 && b_subs.len() == 2 && out_subs.len() == 2 {
        let (ai, aj) = (a_subs[0], a_subs[1]);
        let (bj, bk) = (b_subs[0], b_subs[1]);
        let (oi, ok) = (out_subs[0], out_subs[1]);

        if aj == bj && ai == oi && bk == ok && ai != bk {
            return Ok(Some(client.matmul(a, b)?));
        }

        // "ij,kj->ik" = A @ B^T
        if aj == bk && ai == oi && bj == ok && ai != bj {
            let bt = b.transpose(-1, -2)?;
            return Ok(Some(client.matmul(a, &bt)?));
        }

        // "ji,jk->ik" = A^T @ B
        if ai == bj && aj == oi && bk == ok && aj != bk {
            let at = a.transpose(-1, -2)?;
            return Ok(Some(client.matmul(&at, b)?));
        }
    }

    // Batch matmul: "bij,bjk->bik"
    if a_subs.len() == 3 && b_subs.len() == 3 && out_subs.len() == 3 {
        let (ab, ai, aj) = (a_subs[0], a_subs[1], a_subs[2]);
        let (bb, bj, bk) = (b_subs[0], b_subs[1], b_subs[2]);
        let (ob, oi, ok) = (out_subs[0], out_subs[1], out_subs[2]);

        if ab == bb && ab == ob && aj == bj && ai == oi && bk == ok {
            return Ok(Some(client.matmul(a, b)?));
        }
    }

    Ok(None)
}

/// General binary einsum: broadcast-multiply then sum contracted dims.
fn einsum_binary_general<R, C>(
    client: &C,
    a_subs: &[char],
    b_subs: &[char],
    out_subs: &[char],
    a: &Tensor<R>,
    b: &Tensor<R>,
    label_sizes: &HashMap<char, usize>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + ReduceOps<R> + MatmulOps<R> + ScalarOps<R>,
{
    // Collect all unique labels in a canonical order
    let mut all_labels: Vec<char> = Vec::new();
    for &c in a_subs.iter().chain(b_subs.iter()) {
        if !all_labels.contains(&c) {
            all_labels.push(c);
        }
    }

    let output_set: BTreeSet<char> = out_subs.iter().copied().collect();

    // Reshape A and B to have all_labels dimensions (inserting size-1 for missing dims)
    let a_expanded = expand_to_labels(a, a_subs, &all_labels, label_sizes)?;
    let b_expanded = expand_to_labels(b, b_subs, &all_labels, label_sizes)?;

    // Element-wise multiply (broadcasting handles size-1 dims)
    let product = client.mul(&a_expanded, &b_expanded)?;

    // Sum over contracted dims (not in output)
    let contract_dims: Vec<usize> = all_labels
        .iter()
        .enumerate()
        .filter(|(_, label)| !output_set.contains(label))
        .map(|(dim, _)| dim)
        .collect();

    let mut result = if !contract_dims.is_empty() {
        client.sum(&product, &contract_dims, false)?
    } else {
        product
    };

    // The remaining dims after summing correspond to labels still present
    let remaining_labels: Vec<char> = all_labels
        .iter()
        .enumerate()
        .filter(|(dim, _)| !contract_dims.contains(dim))
        .map(|(_, &label)| label)
        .collect();

    // Permute to match output order if needed
    if remaining_labels != out_subs && !out_subs.is_empty() {
        let perm: Vec<usize> = out_subs
            .iter()
            .map(|label| {
                remaining_labels
                    .iter()
                    .position(|l| l == label)
                    .unwrap_or(0)
            })
            .collect();

        if perm.len() > 1 {
            result = result.permute(&perm)?;
        }
    }

    Ok(result)
}

/// Expand a tensor to have dimensions for all labels.
///
/// Missing labels get size-1 dimensions inserted. Existing dims are permuted
/// to match the canonical label order.
fn expand_to_labels<R: Runtime>(
    tensor: &Tensor<R>,
    tensor_subs: &[char],
    all_labels: &[char],
    label_sizes: &HashMap<char, usize>,
) -> Result<Tensor<R>> {
    // First, permute existing dims to match their order in all_labels
    let existing_positions: Vec<(usize, usize)> = all_labels
        .iter()
        .enumerate()
        .filter_map(|(target_pos, label)| {
            tensor_subs
                .iter()
                .position(|s| s == label)
                .map(|src_pos| (src_pos, target_pos))
        })
        .collect();

    // Permute if needed
    let mut current = tensor.clone();
    let src_order: Vec<usize> = existing_positions.iter().map(|(src, _)| *src).collect();
    let identity: Vec<usize> = (0..src_order.len()).collect();
    if src_order != identity {
        current = current.permute(&src_order)?;
    }

    // Now insert size-1 dims for missing labels
    // After permute, dims are in the order they appear in all_labels (among present labels)
    let present_labels: BTreeSet<char> = tensor_subs.iter().copied().collect();

    for (i, label) in all_labels.iter().enumerate() {
        if !present_labels.contains(label) {
            current = current.unsqueeze(i as isize)?;
        }
    }

    // Broadcast to full sizes
    let target_shape: Vec<usize> = all_labels
        .iter()
        .map(|label| *label_sizes.get(label).unwrap())
        .collect();

    current = current.broadcast_to(&target_shape)?;

    Ok(current)
}

/// N-ary einsum: left-fold pairwise contractions.
///
/// Splits the notation into pairwise operations and reduces left-to-right.
fn einsum_nary<R, C>(client: &C, notation: &str, inputs: &[&Tensor<R>]) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + ReduceOps<R> + MatmulOps<R> + ScalarOps<R>,
{
    let parsed = parse_notation(notation, inputs.len())?;
    let final_output = &parsed.output_subscripts;

    // Left-fold: contract input[0] with input[1], then result with input[2], etc.
    let mut accumulated = inputs[0].clone();
    let mut acc_subs = parsed.input_subscripts[0].clone();

    for i in 1..inputs.len() {
        let next_subs = &parsed.input_subscripts[i];

        // Determine intermediate output: keep all labels needed for remaining contractions
        // or in the final output
        let mut needed_labels: BTreeSet<char> = final_output.iter().copied().collect();
        for j in (i + 1)..inputs.len() {
            for &c in &parsed.input_subscripts[j] {
                needed_labels.insert(c);
            }
        }

        // Intermediate output = labels in acc_subs or next_subs that are in needed_labels
        let intermediate_output: Vec<char> = {
            let mut labels = Vec::new();
            for &c in acc_subs.iter().chain(next_subs.iter()) {
                if needed_labels.contains(&c) && !labels.contains(&c) {
                    labels.push(c);
                }
            }
            labels
        };

        // Build pairwise notation
        let acc_str: String = acc_subs.iter().collect();
        let next_str: String = next_subs.iter().collect();
        let out_str: String = intermediate_output.iter().collect();
        let pair_notation = format!("{acc_str},{next_str}->{out_str}");

        accumulated = einsum_impl(client, &pair_notation, &[&accumulated, inputs[i]])?;
        acc_subs = intermediate_output;
    }

    // Final permute if needed
    if acc_subs != *final_output && !final_output.is_empty() {
        let perm: Vec<usize> = final_output
            .iter()
            .map(|label| acc_subs.iter().position(|l| l == label).unwrap_or(0))
            .collect();
        if perm.len() > 1 {
            accumulated = accumulated.permute(&perm)?;
        }
    }

    Ok(accumulated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_explicit_notation() {
        let parsed = parse_notation("ij,jk->ik", 2).unwrap();
        assert_eq!(
            parsed.input_subscripts,
            vec![vec!['i', 'j'], vec!['j', 'k']]
        );
        assert_eq!(parsed.output_subscripts, vec!['i', 'k']);
    }

    #[test]
    fn test_parse_implicit_notation() {
        let parsed = parse_notation("ij,jk", 2).unwrap();
        assert_eq!(
            parsed.input_subscripts,
            vec![vec!['i', 'j'], vec!['j', 'k']]
        );
        // j appears twice, i and k appear once -> output = [i, k]
        assert_eq!(parsed.output_subscripts, vec!['i', 'k']);
    }

    #[test]
    fn test_parse_trace() {
        let parsed = parse_notation("ii->", 1).unwrap();
        assert_eq!(parsed.input_subscripts, vec![vec!['i', 'i']]);
        assert_eq!(parsed.output_subscripts, Vec::<char>::new());
    }

    #[test]
    fn test_parse_wrong_input_count() {
        assert!(parse_notation("ij,jk->ik", 3).is_err());
    }

    #[test]
    fn test_parse_invalid_subscript() {
        assert!(parse_notation("iJ,jk->ik", 2).is_err());
    }
}
