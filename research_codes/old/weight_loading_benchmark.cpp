// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/functions.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/layout/page_config.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/common/queue_id.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/assert.hpp>
#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace ttnn;

struct BenchmarkConfig {
    int in_features = 4096;
    int out_features = 4096;
    int large_batch_size = 256;
    int small_batch_size = 32;
    int minibatches = 8;
    int warmup_iters = 2;
    int measure_iters = 5;
    int seed = 1337;
    bool run_large_batch = true;
    bool run_mini_batch = true;
};

struct TimingResult {
    double kernel_compilation_ms = 0.0;
    double weight_load_ms = 0.0;
    double forward_compute_ms = 0.0;
    double total_ms = 0.0;
    std::vector<double> individual_forward_times_ms;
};

// Create random tensor on device
Tensor create_random_tensor(Device* device, const Shape& shape, DataType dtype, int seed) {
    // Create tensor from torch (simplified - in real code would use proper initialization)
    // For now, we'll use a placeholder that creates the tensor structure
    // In actual implementation, you would:
    // 1. Generate random data on host
    // 2. Convert to ttnn tensor format
    // 3. Transfer to device

    // This is a simplified version - actual implementation would be more complex
    return Tensor();  // Placeholder
}

// Create weight and bias tensors
std::pair<Tensor, Tensor> create_weight_and_bias(
    Device* device, int in_features, int out_features, DataType dtype, int seed, TimingResult& timings) {
    auto start = std::chrono::high_resolution_clock::now();

    // Create weight tensor: [out_features, in_features]
    Shape weight_shape = {1, 1, out_features, in_features};
    Tensor weight = create_random_tensor(device, weight_shape, dtype, seed);

    // Create bias tensor: [out_features]
    Shape bias_shape = {1, 1, 1, out_features};
    Tensor bias = create_random_tensor(device, bias_shape, dtype, seed + 1);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    timings.weight_load_ms = duration.count() / 1000.0;

    return {weight, bias};
}

// Run large batch benchmark
TimingResult run_large_batch_benchmark(Device* device, const BenchmarkConfig& config) {
    TimingResult result;

    std::cout << "[PROFILE] Large Batch Benchmark (B=" << config.large_batch_size << ")\n";

    // Create weight and bias
    auto [weight, bias] = create_weight_and_bias(
        device, config.in_features, config.out_features, DataType::BFLOAT16, config.seed, result);

    // Measure kernel compilation
    auto compile_start = std::chrono::high_resolution_clock::now();

    // Create linear operation (this compiles kernels on first call)
    Shape input_shape = {1, 1, config.large_batch_size, config.in_features};
    Tensor input = create_random_tensor(device, input_shape, DataType::BFLOAT16, config.seed + 2);

    // First call compiles kernels
    ZoneScopedN("LargeBatch_Compile");
    Tensor output = ttnn::linear(
        input,
        weight,
        bias,
        false,         // transpose_a
        false,         // transpose_b
        std::nullopt,  // memory_config
        std::nullopt,  // dtype
        std::nullopt,  // program_config
        std::nullopt,  // activation
        std::nullopt,  // compute_kernel_config
        std::nullopt,  // core_grid
        std::nullopt,  // output_tile
        std::nullopt,  // optional_output_tensor
        std::nullopt,  // global_cb
        std::nullopt   // sub_device_id
    );

    // Synchronize to ensure compilation is complete
    device->synchronize();

    auto compile_end = std::chrono::high_resolution_clock::now();
    auto compile_duration = std::chrono::duration_cast<std::chrono::microseconds>(compile_end - compile_start);
    result.kernel_compilation_ms = compile_duration.count() / 1000.0;

    std::cout << "[PROFILE] Kernel compilation: " << result.kernel_compilation_ms << " ms\n";

    // Warmup iterations
    for (int i = 0; i < config.warmup_iters; ++i) {
        ZoneScopedN("LargeBatch_Warmup");
        std::cout << "[PROFILE] Warmup forward " << i << "\n";
        output = ttnn::linear(input, weight, bias);
        device->synchronize();
    }

    // Measurement iterations
    std::vector<double> forward_times;
    for (int i = 0; i < config.measure_iters; ++i) {
        ZoneScopedN("LargeBatch_Measure");
        std::cout << "[PROFILE] Measurement forward " << i << "\n";

        auto fwd_start = std::chrono::high_resolution_clock::now();
        {
            ZoneScopedN("LargeBatch_ForwardPass");
            output = ttnn::linear(input, weight, bias);
        }
        device->synchronize();
        auto fwd_end = std::chrono::high_resolution_clock::now();

        auto fwd_duration = std::chrono::duration_cast<std::chrono::microseconds>(fwd_end - fwd_start);
        double fwd_ms = fwd_duration.count() / 1000.0;
        forward_times.push_back(fwd_ms);
    }

    // Calculate averages
    double sum = 0.0;
    for (double t : forward_times) {
        sum += t;
    }
    result.forward_compute_ms = sum / forward_times.size();
    result.individual_forward_times_ms = forward_times;

    // For first iteration, include weight loading
    result.total_ms = result.weight_load_ms + result.forward_compute_ms;

    return result;
}

// Run mini-batch benchmark
TimingResult run_mini_batch_benchmark(Device* device, const BenchmarkConfig& config) {
    TimingResult result;

    std::cout << "[PROFILE] Mini-Batch Benchmark (b=" << config.small_batch_size << ", " << config.minibatches
              << " minibatches)\n";

    // Create weight and bias
    auto [weight, bias] = create_weight_and_bias(
        device, config.in_features, config.out_features, DataType::BFLOAT16, config.seed, result);

    // Create large input tensor for slicing
    Shape large_input_shape = {1, 1, config.small_batch_size * config.minibatches, config.in_features};
    Tensor large_input = create_random_tensor(device, large_input_shape, DataType::BFLOAT16, config.seed + 2);

    // Measure kernel compilation
    auto compile_start = std::chrono::high_resolution_clock::now();

    // Create dummy small input for compilation
    Shape dummy_shape = {1, 1, config.small_batch_size, config.in_features};
    Tensor dummy_input = create_random_tensor(device, dummy_shape, DataType::BFLOAT16, config.seed + 3);

    ZoneScopedN("MiniBatch_Compile");
    Tensor dummy_output = ttnn::linear(dummy_input, weight, bias);
    device->synchronize();

    auto compile_end = std::chrono::high_resolution_clock::now();
    auto compile_duration = std::chrono::duration_cast<std::chrono::microseconds>(compile_end - compile_start);
    result.kernel_compilation_ms = compile_duration.count() / 1000.0;

    std::cout << "[PROFILE] Kernel compilation: " << result.kernel_compilation_ms << " ms\n";

    // Warmup sequences
    for (int seq = 0; seq < config.warmup_iters; ++seq) {
        ZoneScopedN("MiniBatch_WarmupSequence");
        for (int i = 0; i < config.minibatches; ++i) {
            // Slice input (simplified - actual implementation would use proper slicing)
            // For now, we'll use the full large input as a placeholder
            ZoneScopedN("MiniBatch_WarmupForward");
            dummy_output = ttnn::linear(dummy_input, weight, bias);
            device->synchronize();
        }
    }

    // Measurement sequences
    std::vector<double> sequence_times;
    for (int seq = 0; seq < config.measure_iters; ++seq) {
        ZoneScopedN("MiniBatch_MeasureSequence");
        std::cout << "[PROFILE] Measurement sequence " << seq << "\n";

        auto seq_start = std::chrono::high_resolution_clock::now();
        double seq_forward_time = 0.0;

        for (int i = 0; i < config.minibatches; ++i) {
            ZoneScopedN("MiniBatch_ForwardPass");
            std::cout << "[PROFILE]   Forward pass " << i << "\n";

            auto fwd_start = std::chrono::high_resolution_clock::now();
            {
                // Explicit marker for device profiler
                ZoneScopedN("MiniBatch_SingleForward");
                dummy_output = ttnn::linear(dummy_input, weight, bias);
            }
            device->synchronize();
            auto fwd_end = std::chrono::high_resolution_clock::now();

            auto fwd_duration = std::chrono::duration_cast<std::chrono::microseconds>(fwd_end - fwd_start);
            double fwd_ms = fwd_duration.count() / 1000.0;
            seq_forward_time += fwd_ms;
            result.individual_forward_times_ms.push_back(fwd_ms);
        }

        auto seq_end = std::chrono::high_resolution_clock::now();
        auto seq_duration = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - seq_start);
        double seq_ms = seq_duration.count() / 1000.0;
        sequence_times.push_back(seq_ms);
    }

    // Calculate averages
    double sum = 0.0;
    for (double t : sequence_times) {
        sum += t;
    }
    result.forward_compute_ms = sum / sequence_times.size();

    // For first sequence, include weight loading
    result.total_ms = result.weight_load_ms + result.forward_compute_ms;

    return result;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;

    // Parse command line arguments (simplified)
    // In full implementation, would use proper argument parsing

    std::cout << "================================================================================\n";
    std::cout << "WEIGHT LOADING BENCHMARK (C++)\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";
    std::cout << "Configuration:\n";
    std::cout << "  In features: " << config.in_features << "\n";
    std::cout << "  Out features: " << config.out_features << "\n";
    std::cout << "  Large batch size: " << config.large_batch_size << "\n";
    std::cout << "  Small batch size: " << config.small_batch_size << "\n";
    std::cout << "  Minibatches: " << config.minibatches << "\n";
    std::cout << "  Warmup iterations: " << config.warmup_iters << "\n";
    std::cout << "  Measurement iterations: " << config.measure_iters << "\n";
    std::cout << "\n";

    // Initialize device
    // In actual implementation, would properly initialize device
    // For now, this is a template

    std::cout << "NOTE: This is a template C++ benchmark code.\n";
    std::cout << "Full implementation requires:\n";
    std::cout << "  1. Proper tensor creation and initialization\n";
    std::cout << "  2. Device initialization\n";
    std::cout << "  3. Proper tensor slicing for mini-batch scenario\n";
    std::cout << "  4. Integration with device profiler markers\n";
    std::cout << "\n";
    std::cout << "The key improvement over Python code:\n";
    std::cout << "  - Each forward pass is explicitly marked with ZoneScopedN()\n";
    std::cout << "  - Device profiler can accurately identify forward pass boundaries\n";
    std::cout << "  - No need to guess which run_ids belong to same forward pass\n";
    std::cout << "\n";

    return 0;
}
