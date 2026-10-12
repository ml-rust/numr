# Distributed Computing

numr exposes collective communication through the `Communicator` trait in `numr::runtime`.
The trait covers all-reduce, broadcast, all-gather, reduce-scatter, and point-to-point messages.

| Type | Path | Feature | Use |
| ---- | ---- | ------- | --- |
| `Communicator` | `numr::runtime::Communicator` | - | The collective trait |
| `NoOpCommunicator` | `numr::runtime::NoOpCommunicator` | - (always available) | Single-process stand-in |
| `NcclCommunicator` | `numr::runtime::NcclCommunicator` | `nccl` | Multi-GPU collectives through NCCL |
| `NexarNetCommunicator` | `numr::runtime::NexarNetCommunicator` | `distributed` | Multi-machine runs over pure-Rust QUIC transport through [nexar](https://github.com/ml-rust/nexar) |
| `HierarchicalCommunicator` | `numr::runtime::HierarchicalCommunicator` | `distributed-gpu` | NCCL within a node, nexar between nodes |
| `CommunicatorGroup` | `numr::runtime::CommunicatorGroup` | - | Splits a world communicator into sub-groups |
| `BackwardHook` | `numr::autograd::BackwardHook` | - | Triggers cross-node gradient synchronization during `backward()` |

## Parallel Groups

`CommunicatorGroup` splits a world communicator by `ParallelDim`:

| `ParallelDim` | Meaning |
| ------------- | ------- |
| `Data` | Replicate the model, shard the data |
| `Tensor` | Shard weight matrices within a layer |
| `Pipeline` | Shard layers across stages |
| `Expert` | Distribute MoE experts across devices |

## Feature Requirements

Enable the feature for each communicator, as the table shows. `NoOpCommunicator` needs no feature.
`distributed-gpu` enables `distributed` and `nccl`.
