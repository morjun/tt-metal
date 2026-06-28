# Research Proposal: Zero-Overhead In-SRAM Retrieval-Augmented Generation via FPU Spatial Multiplexing

**Author:** Junmo Chung

**Affiliation:** ANLAB, School of Computing, KAIST

## 1. Introduction & Motivation

Retrieval-Augmented Generation (RAG) has become the de facto standard for mitigating hallucinations in Large Language Models (LLMs). However, conventional RAG architectures inherently separate the retrieval mechanism (e.g., CPU/GPU-based vector databases) from the generation pipeline. This physical and logical decoupling incurs severe latency penalties during the decode phase due to PCIe communication overheads and memory bandwidth bottlenecks.

Simultaneously, spatial architectures such as Tenstorrent NPUs suffer from extreme underutilization during the Batch-1 decode stage. While the attention (SDPA) phase effectively saturates the $32 \times 32$ FPU tiles by stacking multiple attention heads, the linear projections (QKV and MLP) process a single token embedding (`[1, Hidden_dim]`). This architectural mismatch results in a 97% hardware idle rate, where 31 out of 32 rows in the FPU tile are zero-padded and wasted.

This proposal introduces **In-SRAM RAG**, a novel hardware-algorithm co-design that completely fuses the retrieval and generation pipelines into a single NPU execution cycle. By spatially multiplexing candidate document vectors into the wasted FPU rows during QKV projection, we achieve zero-overhead, in-core vector retrieval without consuming additional clock cycles or DRAM bandwidth.

## 2. Proposed Architecture

### 2.1. Hierarchical Cascaded Retrieval and SRAM Streaming

To overcome the capacity constraints of the on-chip L1 SRAM (~200MB aggregate), we propose a two-stage hierarchical retrieval system:

1. **Host-Side Coarse Filtering:** A lightweight retrieval algorithm (e.g., BM25 or quantized ANN) on the host CPU identifies a candidate set of Top-1000 document vectors.
2. **On-Chip Streaming:** The candidate vectors are streamed into the abundant, highly-distributed L1 SRAM ring buffers of the Tensix cores, entirely bypassing the device DRAM.

### 2.2. Zero-Overhead FPU Spatial Multiplexing (Piggybacking)

The core innovation lies in repurposing the 31 wasted rows of the FPU matrix engine. Since both the token embedding and the document vectors share identical dimensionality (`[1, Hidden_dim]`), they can be seamlessly concatenated.

* During the QKV linear projection, the scalar engine (NCRISC) fetches 31 document vectors from the L1 SRAM and injects them into Rows 1 through 31 of the input matrix $X$.
* The matrix engine (TRISC) performs the standard projection $Y = X \times W_{qkv}$.
* Due to the mathematical independence of matrix rows, Row 0 of the output $Y$ perfectly yields the LLM's QKV vector, while Rows 1-31 inherently compute the projection of the document vectors into the LLM's latent space, effectively performing an in-core similarity dot-product for re-ranking.

### 2.3. Zero DRAM-Bandwidth RAG

By maintaining the candidate vectors exclusively within the interleaved L1 SRAM and utilizing the NoC (Network-on-Chip) torus for internal routing, this architecture consumes **zero DRAM bandwidth**. This is critical as the linear projection stages in LLM decode are strictly memory-bound by the weight-streaming wall.

## 3. Methodology & Evaluation Plan

### 3.1. Implementation

The proposed architecture will be implemented on the Tenstorrent environment (e.g., Blackhole/Wormhole architectures). We will modify the low-level C++ Tensix kernels to enable the NCRISC to dual-fetch the token embedding and the L1-resident document vectors simultaneously.

### 3.2. Evaluation Metrics

* **End-to-End Latency:** Measure the generation token-per-second (tok/s) with and without In-SRAM RAG to prove the "zero-overhead" hypothesis.
* **Hardware Utilization:** Profile the TRISC FPU active cycles and NCRISC memory fetch overlapping to demonstrate a 100% tile utilization rate during linear projections.
* **Retrieval Accuracy:** Compare the re-ranking accuracy (Recall@K) of the In-SRAM dot-product against conventional isolated vector databases (e.g., Faiss, Milvus).

## 4. Expected Contributions

1. **First Spatial Fusion of RAG and LLM:** Proposing an architecture that natively integrates vector retrieval into the NPU's linear projection pipeline.
2. **100% FPU Utilization in Batch-1 Decode:** Resolving the persistent 31/32 row waste in spatial accelerators through algorithmic multiplexing rather than hardware modifications.
3. **DRAM-less Retrieval:** Demonstrating that high-quality RAG can be achieved with zero additional DRAM bandwidth penalty, maximizing the theoretical throughput of memory-bound LLM serving systems.
