use crate::runtime::cpu::CpuRuntime;
use crate::tensor::Tensor;

/// Raw bit pattern of every element — no tolerance, no rounding.
pub(super) fn bits(tensor: &Tensor<CpuRuntime>) -> Vec<u32> {
    tensor.to_vec::<f32>().iter().map(|v| v.to_bits()).collect()
}
