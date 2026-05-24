// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode.hpp"

#include <utility>

#include "device/sdpa_decode_op.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace {
inline uint32_t get_chunk_size(uint32_t s) {
    /*
    # find maximum power of 2 divisor of s
    for i in range(1, s):
        if s % (2**(i+1)) != 0:
            break
    */
    uint32_t i = 1;
    for (; i < s; i++) {
        if (s % (1 << (i + 1)) != 0) {
            break;
        }
    }
    return std::min(512, 1 << i);
}
}  // namespace

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::vector<uint32_t>& cur_pos,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t l1_sink_size,
    float l1_min_expected_hit_ratio,
    const std::optional<const Tensor>& l1_k_tensor,
    const std::optional<const Tensor>& l1_v_tensor,
    const std::vector<std::optional<const Tensor>>& l1_k_tensors,
    const std::vector<std::optional<const Tensor>>& l1_v_tensors,
    const std::vector<uint32_t>& l1_tier_token_starts,
    const std::vector<uint32_t>& l1_tier_token_counts,
    uint32_t l1_decode_start_pos,
    bool l1_only_mode) {
    [[maybe_unused]] auto arch =
        input_tensor_q.storage_type() == StorageType::DEVICE
            ? input_tensor_q.device()->arch()
            : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    uint32_t s = input_tensor_k.logical_shape()[-2];
    uint32_t k_chunk_size = get_chunk_size(s);
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    } else {
        TT_FATAL(
            k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: {}",
            k_chunk_size);
    }

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    // Determine effective l1_k/v tensors (tier 0): if tier vector is provided, use its first element.
    auto effective_l1_k = !l1_k_tensors.empty() ? l1_k_tensors[0] : l1_k_tensor;
    auto effective_l1_v = !l1_v_tensors.empty() ? l1_v_tensors[0] : l1_v_tensor;

    // Build optional_inputs: [cur_pos, page_table(null), attn_mask, attention_sink, l1_k_0, l1_v_0, l1_k_1, l1_v_1,
    // ...]
    std::vector<std::optional<const Tensor>> optional_inputs = {
        cur_pos_tensor, std::nullopt, attn_mask, attention_sink, effective_l1_k, effective_l1_v};
    // Pack additional tiers (index 1..N-1) at [6, 7, 8, 9, ...]
    for (size_t i = 1; i < l1_k_tensors.size(); ++i) {
        optional_inputs.push_back(l1_k_tensors[i]);
        optional_inputs.push_back(l1_v_tensors[i]);
    }

    return operation::run(
               ScaledDotProductAttentionDecode{
                   .is_causal = is_causal,
                   .cur_pos = cur_pos,
                   .scale = scale,
                   .sliding_window_size = sliding_window_size,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = program_config,
                   .compute_kernel_config = kernel_config_val,
                   .k_chunk_size = k_chunk_size,
                   .paged_attention = false,
                   .l1_sink_size = l1_sink_size,
                   .l1_min_expected_hit_ratio = l1_min_expected_hit_ratio,
                   .l1_tier_token_starts = l1_tier_token_starts,
                   .l1_tier_token_counts = l1_tier_token_counts,
                   .l1_decode_start_pos = l1_decode_start_pos,
                   .l1_only_mode = l1_only_mode},
               {input_tensor_q, input_tensor_k, input_tensor_v},
               optional_inputs,
               {})
        .at(0);
}

ttnn::Tensor ExecutePagedScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t l1_sink_size,
    float l1_min_expected_hit_ratio) {
    [[maybe_unused]] auto arch =
        input_tensor_q.storage_type() == StorageType::DEVICE
            ? input_tensor_q.device()->arch()
            : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();

    // Use k_chunk_size as override; if k_chunk_size == 0, figure it out in kernels
    // uint32_t k_chunk_size = get_chunk_size(s);
    uint32_t k_chunk_size = 0;
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    }

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               ScaledDotProductAttentionDecode{
                   .is_causal = is_causal,
                   .cur_pos = std::vector<uint32_t>(),
                   .scale = scale,
                   .sliding_window_size = sliding_window_size,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = program_config,
                   .compute_kernel_config = kernel_config_val,
                   .k_chunk_size = k_chunk_size,
                   .paged_attention = true,
                   .l1_sink_size = l1_sink_size,
                   .l1_min_expected_hit_ratio = l1_min_expected_hit_ratio},
               {input_tensor_q, input_tensor_k, input_tensor_v},
               {cur_pos_tensor, page_table_tensor, attn_mask, attention_sink},
               {})
        .at(0);
}

ttnn::Tensor ExecuteFlashMultiLatentAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::vector<uint32_t>& cur_pos,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch =
        input_tensor_q.storage_type() == StorageType::DEVICE
            ? input_tensor_q.device()->arch()
            : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    uint32_t s = input_tensor_k.logical_shape()[-2];
    uint32_t k_chunk_size = get_chunk_size(s);
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    } else {
        TT_FATAL(
            k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: {}",
            k_chunk_size);
    }

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               ScaledDotProductAttentionDecode{
                   .is_causal = is_causal,
                   .cur_pos = cur_pos,
                   .scale = scale,
                   .sliding_window_size = sliding_window_size,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = program_config,
                   .compute_kernel_config = kernel_config_val,
                   .k_chunk_size = k_chunk_size,
                   .paged_attention = false,
                   .l1_sink_size = 0,
                   .l1_min_expected_hit_ratio = 0.0f,
                   .use_mla = true,
                   .head_dim_v = head_dim_v},
               {input_tensor_q, input_tensor_k},
               {cur_pos_tensor, std::nullopt, attn_mask, attention_sink},
               {})
        .at(0);
}

ttnn::Tensor ExecutePagedFlashMultiLatentAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    const bool is_causal,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& attention_sink,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    [[maybe_unused]] auto arch =
        input_tensor_q.storage_type() == StorageType::DEVICE
            ? input_tensor_q.device()->arch()
            : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();

    // Use k_chunk_size as override; if k_chunk_size == 0, figure it out in kernels
    // uint32_t k_chunk_size = get_chunk_size(s);
    uint32_t k_chunk_size = 0;
    if (program_config.has_value() && program_config.value().k_chunk_size > 0) {
        k_chunk_size = program_config.value().k_chunk_size;
        // assert chunk size must be power of 2 and multiple of 32
        TT_FATAL(
            (k_chunk_size & (k_chunk_size - 1)) == 0,
            "User provided k_chunk_size must be power of 2, got: {}",
            k_chunk_size);
        TT_FATAL(k_chunk_size % 32 == 0, "User provided k_chunk_size must be multiple of 32, got: {}", k_chunk_size);
    }

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               ScaledDotProductAttentionDecode{
                   .is_causal = is_causal,
                   .cur_pos = std::vector<uint32_t>(),
                   .scale = scale,
                   .sliding_window_size = sliding_window_size,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = program_config,
                   .compute_kernel_config = kernel_config_val,
                   .k_chunk_size = k_chunk_size,
                   .paged_attention = true,
                   .l1_sink_size = 0,
                   .l1_min_expected_hit_ratio = 0.0f,
                   .use_mla = true,
                   .head_dim_v = head_dim_v},
               {input_tensor_q, input_tensor_k},
               {cur_pos_tensor, page_table_tensor, attn_mask, attention_sink},
               {})
        .at(0);
}

}  // namespace ttnn::operations::transformer
