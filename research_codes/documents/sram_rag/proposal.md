# Research Proposal: Zero-Overhead In-SRAM Retrieval-Augmented Generation via FPU Spatial Multiplexing

**Author:** Junmo Chung

**Affiliation:** ANLAB, School of Computing, KAIST

## 1. Introduction & Motivation

Retrieval-Augmented Generation (RAG) has become the de facto standard for mitigating hallucinations in Large Language Models (LLMs). However, conventional RAG architectures inherently separate the retrieval mechanism (e.g., CPU/GPU-based vector databases) from the generation pipeline. This physical and logical decoupling incurs severe latency penalties during the decode phase due to PCIe communication overheads and memory bandwidth bottlenecks.

Simultaneously, spatial architectures such as Tenstorrent NPUs suffer from extreme underutilization during the Batch-1 decode stage. While the attention (SDPA) phase effectively saturates the $32 \times 32$ FPU tiles by stacking multiple attention heads, the linear projections (QKV and MLP) process a single token embedding (`[1, Hidden_dim]`). This architectural mismatch results in a 97% hardware idle rate, where 31 out of 32 rows in the FPU tile are zero-padded and wasted.

This proposal introduces **In-SRAM RAG**, a novel hardware-algorithm co-design that completely fuses the retrieval and generation pipelines into a single NPU execution cycle. By spatially multiplexing candidate document vectors into the wasted FPU rows during QKV projection, we achieve zero-overhead, in-core vector retrieval without consuming additional clock cycles or DRAM bandwidth.

## 2. Proposed Architecture

### 2.1. Distributed SRAM Streaming (Interleaved Topology)

To overcome the capacity constraints of the on-chip L1 SRAM and avoid sequential loop overheads, we utilize an interleaved memory topology.

1. **Coarse Filtering:** A lightweight retrieval algorithm on the host CPU identifies a candidate set of Top-1000 document vectors.
2. **Distributed Placement:** The 1000 candidate vectors are broadcasted and evenly interleaved across the ~120 Tensix cores via the NoC torus. Consequently, each core’s L1 SRAM holds only 8~9 document vectors, fitting perfectly within a single 32-row FPU tile limit.

### 2.2. Zero-Loop FPU Spatial Multiplexing (Piggybacking)

Because each core is responsible for only 8~9 documents, the entire 1000-document QKV projection is resolved in a **single matrix multiplication cycle** without iterative looping.

* During the QKV linear projection, the input matrix $X$ is constructed as: Row 0 (current token), Rows 1-9 (the core's assigned document vectors), and Rows 10-31 (zero-padded).
* The matrix engine (TRISC) performs the standard projection $Y = X \times W_{qkv}$.
* Due to the mathematical independence of matrix rows, Row 0 of the output perfectly yields the LLM's QKV vector, while Rows 1-9 inherently compute the projection of the document vectors into the LLM's latent space.

### 2.3. Zero DRAM-Bandwidth RAG

By maintaining the candidate vectors exclusively within the interleaved L1 SRAM and utilizing the NoC for internal routing, this architecture consumes **zero DRAM bandwidth**. This is critical as the linear projection stages in LLM decode are strictly memory-bound by the weight-streaming wall.

### 2.4. Nanosecond-Scale Asynchronous Thresholding & Micro-Rollback

Because token generation and document evaluation occur simultaneously, a 1-token pipeline bubble structurally emerges. To prevent pipeline stalls (Stalls) while evaluating RAG scores, we exploit the heterogeneous asynchronous processors (TRISC and BRISC) inside the Tensix core:

1. **RAG Dot Product (TRISC, ~100 ns):** The FPU computes the dot product of the token's $Q$ and the 8-9 documents' $K$ within a single tile MAC operation, pushing the scalar scores to L1 SRAM.
2. **Main SDPA Compute (TRISC, ~8,900 ns):** The FPU immediately proceeds to the main LLM attention compute (QK^T and PV matmuls over the entire KV cache). This phase is heavily compute-bound, taking roughly 8,900 ns.
3. **Asynchronous Thresholding (BRISC, ~50 ns):** While the FPU is locked in the 8,900 ns SDPA compute, the scalar core (BRISC) asynchronously reads the 8-9 RAG scores and performs a threshold check `if (max_score > threshold)`. This lightweight scalar operation takes less than 50 ns.
4. **Zero-Stall Commit or Micro-Rollback:** * **Miss (99% of cases):** The BRISC flag is 0. The FPU finishes the SDPA, accepts the generated token, and proceeds to the next step with zero delay. The 150 ns RAG overhead is **100% hidden** behind the 8,900 ns SDPA compute.
* **Hit (1% of cases):** The BRISC flag is 1. A **Micro-Rollback** is triggered: the generated token is discarded, the retrieved document's $K, V$ vectors are appended to the KV cache, and step $t$ is recomputed once.



Because high-confidence RAG hits are sparse, the latency cost of this single-step rollback is heavily amortized, ensuring structural correctness with near-zero average latency overhead and absolute zero pipeline stalls.

## 3. Methodology & Evaluation Plan

### 3.1. Implementation

The proposed architecture will be implemented on the Tenstorrent Blackhole architecture. We will modify the low-level C++ Tensix kernels to enable the NCRISC to dual-fetch the token embedding and the distributed L1-resident document vectors simultaneously.

### 3.2. Evaluation Metrics

* **End-to-End Latency:** Measure the generation token-per-second (tok/s) with and without In-SRAM RAG to prove the "100% hidden zero-overhead" hypothesis.
* **Hardware Utilization:** Profile the TRISC FPU active cycles and BRISC thresholding overlap to demonstrate complete pipeline latency hiding.
* **Retrieval Accuracy (Recall@K):** Compare the re-ranking accuracy of the in-core distributed dot-product against conventional isolated vector databases (e.g., Faiss, Milvus) to validate the functional integrity of the multiplexed projection.

## 4. Expected Contributions

1. **First Spatial Fusion of RAG and LLM:** Natively integrating vector retrieval into the NPU's linear projection pipeline.
2. **100% FPU Utilization in Batch-1 Decode:** Resolving the persistent 31/32 row waste in spatial accelerators through algorithmic multiplexing and distributed L1 placement.
3. **Stall-Free Asynchronous RAG:** Proving mathematically and architecturally that in-core document thresholding can be 100% hidden behind the compute-bound SDPA phase, achieving high-precision RAG with zero latency penalty and zero DRAM bandwidth cost.
