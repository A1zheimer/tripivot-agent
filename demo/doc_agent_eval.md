# Distributed Inference Systems for Large Language Models

## Introduction

Large language models have grown beyond the memory capacity of single devices.
Distributed inference systems address this challenge by partitioning model
parameters, activations, and KV cache across multiple accelerators. This
report reviews three representative systems and their trade-offs.

## Pipeline Parallelism

Pipeline parallelism divides the layers of a transformer into stages. Each
stage runs on a separate device, and micro-batches flow through the pipeline
like an assembly line. The main drawback is the pipeline bubble: at the start
and end of each forward pass, some devices remain idle. Bubble ratio can be
reduced by increasing the number of micro-batches, which raises scheduling
complexity. In production clusters, pipeline parallelism is usually combined
with tensor parallelism to balance memory and throughput.

## Tensor Parallelism

Tensor parallelism splits individual weight matrices across devices. For a
feed-forward layer, the first matrix is partitioned by columns and the second
by rows, so only one all-reduce is needed per layer. Communication volume is
therefore proportional to the hidden size rather than the parameter count.
Tensor parallelism requires high-bandwidth interconnects such as NVLink, and
it is typically confined to a single node. For the KV cache, tensor
parallelism shards attention heads so that long contexts benefit directly
from additional devices.

## KV Cache Management

The key-value cache dominates memory consumption during long generation.
Paged attention manages the KV cache in fixed-size blocks, reducing internal
fragmentation from over sixty percent to under four percent. Prefix caching
further exploits shared prompts: when many requests share a system prompt,
the corresponding KV blocks are computed once and reused. Quantized KV cache,
for example eight-bit floating point, halves the memory footprint with minor
quality degradation. Eviction policies such as least-recently-used decide
which blocks to drop when memory pressure rises.

## Scheduling and Batching

Continuous batching replaces static batching in modern serving engines. The
scheduler admits new requests at every decoding step and retires finished
ones immediately, which improves GPU utilization substantially. Under load,
the scheduler must balance latency-sensitive requests against throughput.
Some engines predict the output length to group requests of similar size,
reducing padding waste in the batch.

## Quantization

Weight-only quantization reduces memory traffic while keeping activation
precision intact. Four-bit group quantization is the most common setting,
where weights are stored in four-bit integers and a small floating-point
scale is applied per group. Quantization-aware training recovers part of the
accuracy lost by post-training quantization, at the cost of additional
training compute. For inference clusters, quantization directly translates
into higher batch capacity and lower cost per token.

## Conclusion

Distributed inference is a systems problem as much as an algorithms problem.
Pipeline parallelism, tensor parallelism, KV cache management, continuous
batching, and quantization each attack a different bottleneck. Practical
engines combine all of them, and the best configuration depends on model
size, context length, and the latency budget of the application.
