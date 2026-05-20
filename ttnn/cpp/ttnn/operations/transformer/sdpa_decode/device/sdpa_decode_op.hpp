// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer {

struct ScaledDotProductAttentionDecode {
    const bool is_causal;
    std::vector<uint32_t> cur_pos;
    const std::optional<float> scale;
    const std::optional<uint32_t> sliding_window_size;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<SDPAProgramConfig> program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    const uint32_t k_chunk_size;
    const bool paged_attention;
    const std::optional<bool> share_cache;
    const uint32_t l1_sink_size;
    const float l1_min_expected_hit_ratio;
    const std::vector<uint32_t> l1_tier_token_starts;  // absolute token start per tier
    const std::vector<uint32_t> l1_tier_token_counts;  // token count per tier
    const uint32_t l1_decode_start_pos;                // cur_pos at the first decode step (lower bound for L1 reads)

    const std::optional<bool> use_mla;
    const std::optional<uint32_t> head_dim_v;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    tt::tt_metal::operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

}  // namespace ttnn::operations::transformer
