use super::topo::TopoEntry;
use crate::runtime::Runtime;
use crate::tensor::TensorId;
use std::collections::HashSet;

/// Which gradients the backward driver must produce.
pub(super) enum Wanted {
    /// Every node in the graph — the historical `backward` behavior.
    All,
    /// Only these ids, plus whatever lies on a path to them.
    Set(HashSet<TensorId>),
}

/// Mark every node whose input cone contains a wanted id.
///
/// `topo_order` lists inputs before outputs, so a single forward sweep suffices:
/// `needed[n] = (n in wanted) || any(needed[i] for i in inputs(n))`.
///
/// Returns `None` under [`Wanted::All`], meaning "every node is needed" — no set
/// is built and no lookup is paid.
///
/// This works on `TensorId`s only; it touches no tensor data and is O(nodes + edges).
pub(super) fn compute_needed<R: Runtime>(
    topo_order: &[TopoEntry<R>],
    wanted: &Wanted,
) -> Option<HashSet<TensorId>> {
    let wanted = match wanted {
        Wanted::All => return None,
        Wanted::Set(set) => set,
    };

    let mut needed: HashSet<TensorId> = HashSet::new();
    for (id, _, input_ids) in topo_order {
        if wanted.contains(id) || input_ids.iter().any(|input| needed.contains(input)) {
            needed.insert(*id);
        }
    }
    Some(needed)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::bits;
    use super::*;
    use crate::autograd::{
        Var, backward, backward_wrt, var_add, var_matmul, var_mul, var_sum, var_transpose,
    };
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_backward_wrt_drops_frozen_operand_gradient() {
        // Mirrors a Linear layer on a frozen weight: `var_transpose(&w)` gives
        // an untracked Var, so MatmulBackward records `TensorId::UNTRACKED` for
        // it and no backward mode stores a gradient for the weight.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Trainable activation (stands in for the LoRA-carrying input)
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
            true,
        );
        // Frozen weight
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0, 6.0], &[2, 2], &device).unwrap(),
            false,
        );

        let w_t = var_transpose(&w).unwrap();
        let y = var_matmul(&x, &w_t, &client).unwrap();
        let loss = var_sum(&y, &[0, 1], false, &client).unwrap();

        // Plain backward stores no entry for the frozen operand either
        let full = backward(&loss, &client).unwrap();
        assert!(
            !full.contains(w_t.id()) && !full.contains(TensorId::UNTRACKED),
            "backward must not store a gradient for the transposed frozen weight"
        );

        // Pruned: the unreadable entry is never created
        let pruned = backward_wrt(&loss, &[x.id()], &client).unwrap();
        assert!(
            !pruned.contains(w_t.id()),
            "backward_wrt must not store a gradient for the frozen operand"
        );

        // Every wanted gradient is present and bit-identical
        let want = full.get(x.id()).expect("baseline grad for x");
        let got = pruned.get(x.id()).expect("pruned grad for x");
        assert_eq!(bits(want), bits(got));
    }

    #[test]
    fn test_backward_wrt_bit_identical_to_backward() {
        // Multi-layer graph with a branch: `h` feeds two consumers, so its
        // gradient is accumulated from two terms.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.5f32, -2.25], &[2], &device).unwrap(),
            true,
        );
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.3f32, 0.7], &[2], &device).unwrap(),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.1f32, -0.4], &[2], &device).unwrap(),
            true,
        );

        let build = || {
            let h = var_mul(&x, &a, &client).unwrap();
            let p = var_mul(&h, &b, &client).unwrap();
            let q = var_mul(&h, &h, &client).unwrap();
            let s = var_add(&p, &q, &client).unwrap();
            var_sum(&s, &[0], false, &client).unwrap()
        };

        let full = backward(&build(), &client).unwrap();
        let all_ids = [x.id(), a.id(), b.id()];
        let pruned = backward_wrt(&build(), &all_ids, &client).unwrap();

        for id in all_ids {
            let want = full.get(id).expect("baseline grad");
            let got = pruned.get(id).expect("pruned grad");
            assert_eq!(bits(want), bits(got), "gradient differs for {id:?}");
        }
    }

    #[test]
    fn test_backward_wrt_accumulates_both_diamond_branches() {
        // Diamond: `w` reaches the loss through a short branch and a long one.
        // Both contributions must survive pruning.
        //   left  = w * c1 ; left2 = left * c3   (long branch)
        //   right = w * c2                       (short branch)
        //   loss  = sum(left2 + right)
        //   dL/dw = c1*c3 + c2
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap(),
            true,
        );
        let c1 = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
            false,
        );
        let c2 = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device).unwrap(),
            false,
        );
        let c3 = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
            false,
        );

        let left = var_mul(&w, &c1, &client).unwrap();
        let left2 = var_mul(&left, &c3, &client).unwrap();
        let right = var_mul(&w, &c2, &client).unwrap();
        let s = var_add(&left2, &right, &client).unwrap();
        let loss = var_sum(&s, &[0], false, &client).unwrap();

        let grads = backward_wrt(&loss, &[w.id()], &client).unwrap();
        let grad_w: Vec<f32> = grads.get(w.id()).expect("grad for w").to_vec();

        // 2*3 + 5 = 11. Losing the long branch would give 5.
        assert_eq!(grad_w, vec![11.0f32]);
    }
}
