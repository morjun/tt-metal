// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::operations::transformer::detail {

tt::tt_metal::operation::ProgramWithCallbacks sdpa_decode_multi_core(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    std::optional<const Tensor> cur_pos_tensor,
    std::optional<const Tensor> page_table_tensor,
    std::optional<const Tensor> attn_mask,
    std::optional<const Tensor> attention_sink,
    const Tensor& output_tensor,
    bool is_causal,
    const std::vector<uint32_t>& cur_pos_ids,
    std::optional<float> scale,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config,
    uint32_t k_chunk_size,
    std::optional<bool> share_cache,
    bool mla = false,
    uint32_t head_dim_v = 0,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    uint32_t l1_sink_size = 0,
    float l1_min_expected_hit_ratio = 0.0f,
    std::optional<const Tensor> l1_k_tensor = std::nullopt,
    std::optional<const Tensor> l1_v_tensor = std::nullopt);

}  // namespace ttnn::operations::transformer::detail
