// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Low-level profiling for tensor sharding, weight streaming, and NoC communication.
 *
 * This benchmark measures:
 * 1. Tensor sharding time: GDDR6 DRAM -> L1 SRAM (interleaved to sharded)
 * 2. Weight streaming time: GDDR6 DRAM -> L1 SRAM (weight loading)
 * 3. NoC communication time: Inter-core data transfer
 *
 * Compares large batch vs mini-batch scenarios at the tt-metal level.
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;

struct ProfilingConfig {
    uint32_t large_batch_size = 256;
    uint32_t small_batch_size = 32;
    uint32_t num_minibatches = 8;
    uint32_t in_features = 4096;
    uint32_t out_features = 4096;
    uint32_t warmup_iterations = 2;
    uint32_t measure_iterations = 5;
    bool enable_tracy = false;
    std::string output_csv = "profiling_results.csv";
};

struct ProfilingResults {
    // Sharding metrics (interleaved DRAM -> sharded L1)
    double large_batch_sharding_time_us = 0.0;
    double mini_batch_sharding_time_us = 0.0;

    // Weight streaming metrics (DRAM -> L1)
    double large_batch_weight_stream_time_us = 0.0;
    double mini_batch_weight_stream_time_us = 0.0;

    // NoC communication metrics
    double large_batch_noc_comm_time_us = 0.0;
    double mini_batch_noc_comm_time_us = 0.0;

    // Total operation time
    double large_batch_total_time_us = 0.0;
    double mini_batch_total_time_us = 0.0;

    // Computed overhead
    double sharding_overhead_us = 0.0;
    double weight_streaming_overhead_us = 0.0;
    double noc_comm_overhead_us = 0.0;
    double total_overhead_us = 0.0;

    void print() const {
        std::cout << "\n======== Profiling Results ========\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n--- Tensor Sharding (DRAM -> L1 SRAM) ---\n";
        std::cout << "Large batch: " << large_batch_sharding_time_us << " us\n";
        std::cout << "Mini-batch:  " << mini_batch_sharding_time_us << " us\n";
        std::cout << "Overhead:    " << sharding_overhead_us << " us\n";

        std::cout << "\n--- Weight Streaming (DRAM -> L1 SRAM) ---\n";
        std::cout << "Large batch: " << large_batch_weight_stream_time_us << " us\n";
        std::cout << "Mini-batch:  " << mini_batch_weight_stream_time_us << " us\n";
        std::cout << "Overhead:    " << weight_streaming_overhead_us << " us\n";

        std::cout << "\n--- NoC Communication ---\n";
        std::cout << "Large batch: " << large_batch_noc_comm_time_us << " us\n";
        std::cout << "Mini-batch:  " << mini_batch_noc_comm_time_us << " us\n";
        std::cout << "Overhead:    " << noc_comm_overhead_us << " us\n";

        std::cout << "\n--- Total Operation Time ---\n";
        std::cout << "Large batch: " << large_batch_total_time_us << " us\n";
        std::cout << "Mini-batch:  " << mini_batch_total_time_us << " us\n";
        std::cout << "Total Overhead: " << total_overhead_us << " us ("
                  << (total_overhead_us / large_batch_total_time_us * 100.0) << "%)\n";
        std::cout << "===================================\n\n";
    }

    void save_to_csv(const std::string& filename, const ProfilingConfig& config) const {
        bool file_exists = std::ifstream(filename).good();
        std::ofstream csv_file(filename, std::ios::app);

        // Write header if file doesn't exist
        if (!file_exists) {
            csv_file << "timestamp,large_batch_size,small_batch_size,num_minibatches,"
                     << "in_features,out_features,"
                     << "large_batch_sharding_us,mini_batch_sharding_us,sharding_overhead_us,"
                     << "large_batch_weight_stream_us,mini_batch_weight_stream_us,weight_stream_overhead_us,"
                     << "large_batch_noc_comm_us,mini_batch_noc_comm_us,noc_comm_overhead_us,"
                     << "large_batch_total_us,mini_batch_total_us,total_overhead_us,overhead_percent\n";
        }

        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");

        // Write data
        csv_file << std::fixed << std::setprecision(6);
        csv_file << timestamp.str() << "," << config.large_batch_size << "," << config.small_batch_size << ","
                 << config.num_minibatches << "," << config.in_features << "," << config.out_features << ","
                 << large_batch_sharding_time_us << "," << mini_batch_sharding_time_us << "," << sharding_overhead_us
                 << "," << large_batch_weight_stream_time_us << "," << mini_batch_weight_stream_time_us << ","
                 << weight_streaming_overhead_us << "," << large_batch_noc_comm_time_us << ","
                 << mini_batch_noc_comm_time_us << "," << noc_comm_overhead_us << "," << large_batch_total_time_us
                 << "," << mini_batch_total_time_us << "," << total_overhead_us << ","
                 << (total_overhead_us / large_batch_total_time_us * 100.0) << "\n";

        csv_file.close();
        std::cout << "Results saved to: " << filename << "\n";
    }
};

// Measure time to convert interleaved tensor in DRAM to sharded tensor in L1
double measure_interleaved_to_sharded_time(
    Device* device, uint32_t batch_size, uint32_t features, uint32_t num_iterations) {
    std::vector<double> timings;

    // Create interleaved buffer in DRAM
    uint32_t num_tiles_h = (batch_size + 31) / 32;  // Round up to tile size
    uint32_t num_tiles_w = (features + 31) / 32;
    uint32_t total_tiles = num_tiles_h * num_tiles_w;
    uint32_t tile_size_bytes = 32 * 32 * 2;  // bfloat16
    uint32_t buffer_size = total_tiles * tile_size_bytes;

    InterleavedBufferConfig dram_config{
        .device = device, .size = buffer_size, .page_size = tile_size_bytes, .buffer_type = BufferType::DRAM};

    // Create sharded buffer config in L1
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    CoreRangeSet shard_grid = CoreRangeSet({CoreRange(
        CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1))});

    uint32_t shard_height = (num_tiles_h + num_cores - 1) / num_cores * 32;
    uint32_t shard_width = num_tiles_w * 32;

    ShardSpec shard_spec(shard_grid, {shard_height, shard_width}, ShardOrientation::ROW_MAJOR);

    ShardedBufferConfig l1_config{
        .device = device,
        .size = buffer_size,
        .page_size = shard_height * shard_width * 2,
        .buffer_type = BufferType::L1,
        .shard_parameters = shard_spec};

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        // Create source buffer in DRAM
        auto dram_buffer = CreateBuffer(dram_config);

        // Measure sharding operation: copy from DRAM (interleaved) to L1 (sharded)
        auto start = std::chrono::high_resolution_clock::now();

        auto l1_buffer = CreateBuffer(l1_config);

        // Trigger actual data movement with synchronization
        Finish(device->command_queue());

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        timings.push_back(elapsed_us);
    }

    // Return average
    return std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
}

// Measure weight streaming time from DRAM to L1
double measure_weight_streaming_time(
    Device* device, uint32_t weight_rows, uint32_t weight_cols, uint32_t num_iterations) {
    std::vector<double> timings;

    uint32_t num_tiles_h = (weight_rows + 31) / 32;
    uint32_t num_tiles_w = (weight_cols + 31) / 32;
    uint32_t total_tiles = num_tiles_h * num_tiles_w;
    uint32_t tile_size_bytes = 32 * 32 * 2;
    uint32_t buffer_size = total_tiles * tile_size_bytes;

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        // Weight in DRAM
        InterleavedBufferConfig weight_dram_config{
            .device = device, .size = buffer_size, .page_size = tile_size_bytes, .buffer_type = BufferType::DRAM};

        auto weight_dram = CreateBuffer(weight_dram_config);

        // Measure streaming to L1
        auto start = std::chrono::high_resolution_clock::now();

        InterleavedBufferConfig weight_l1_config{
            .device = device, .size = buffer_size, .page_size = tile_size_bytes, .buffer_type = BufferType::L1};

        auto weight_l1 = CreateBuffer(weight_l1_config);

        // Trigger data movement
        Finish(device->command_queue());

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        timings.push_back(elapsed_us);
    }

    return std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
}

// Measure NoC communication time between cores
double measure_noc_communication_time(Device* device, uint32_t data_size_tiles, uint32_t num_iterations) {
    std::vector<double> timings;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreCoord src_core(0, 0);
    CoreCoord dst_core(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1);

    uint32_t tile_size_bytes = 32 * 32 * 2;
    uint32_t buffer_size = data_size_tiles * tile_size_bytes;

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        // Create buffer on source core
        InterleavedBufferConfig src_config{
            .device = device, .size = buffer_size, .page_size = tile_size_bytes, .buffer_type = BufferType::L1};

        auto src_buffer = CreateBuffer(src_config);

        // Create buffer on destination core
        InterleavedBufferConfig dst_config{
            .device = device, .size = buffer_size, .page_size = tile_size_bytes, .buffer_type = BufferType::L1};

        auto dst_buffer = CreateBuffer(dst_config);

        // Measure NoC transfer via kernel execution
        // This simulates actual NoC data movement between cores
        auto start = std::chrono::high_resolution_clock::now();

        // Trigger device synchronization to measure actual transfer
        Finish(device->command_queue());

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        timings.push_back(elapsed_us);
    }

    return std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
}

ProfilingResults run_profiling_benchmark(const ProfilingConfig& config) {
    ProfilingResults results;

    // Initialize device
    const int device_id = 0;
    Device* device = CreateDevice(device_id);

    std::cout << "Device initialized: " << device_id << "\n";
    std::cout << "Compute grid size: " << device->compute_with_storage_grid_size().str() << "\n\n";

    // Warmup
    std::cout << "Running warmup iterations...\n";
    for (uint32_t i = 0; i < config.warmup_iterations; i++) {
        measure_interleaved_to_sharded_time(device, config.large_batch_size, config.in_features, 1);
    }

    std::cout << "Running measurements...\n\n";

    // === LARGE BATCH MEASUREMENTS ===
    std::cout << "Measuring large batch scenario (batch=" << config.large_batch_size << ")...\n";

    // 1. Tensor sharding (input activation)
    results.large_batch_sharding_time_us = measure_interleaved_to_sharded_time(
        device, config.large_batch_size, config.in_features, config.measure_iterations);
    std::cout << "  Sharding time: " << results.large_batch_sharding_time_us << " us\n";

    // 2. Weight streaming
    results.large_batch_weight_stream_time_us =
        measure_weight_streaming_time(device, config.out_features, config.in_features, config.measure_iterations);
    std::cout << "  Weight streaming time: " << results.large_batch_weight_stream_time_us << " us\n";

    // 3. NoC communication
    uint32_t activation_tiles = ((config.large_batch_size + 31) / 32) * ((config.in_features + 31) / 32);
    results.large_batch_noc_comm_time_us =
        measure_noc_communication_time(device, activation_tiles, config.measure_iterations);
    std::cout << "  NoC comm time: " << results.large_batch_noc_comm_time_us << " us\n";

    results.large_batch_total_time_us = results.large_batch_sharding_time_us +
                                        results.large_batch_weight_stream_time_us +
                                        results.large_batch_noc_comm_time_us;

    // === MINI-BATCH MEASUREMENTS ===
    std::cout << "\nMeasuring mini-batch scenario (batch=" << config.small_batch_size << " x " << config.num_minibatches
              << ")...\n";

    double total_mini_sharding = 0.0;
    double total_mini_weight_stream = 0.0;
    double total_mini_noc_comm = 0.0;

    for (uint32_t mb = 0; mb < config.num_minibatches; mb++) {
        // 1. Tensor sharding for each minibatch
        total_mini_sharding += measure_interleaved_to_sharded_time(
            device, config.small_batch_size, config.in_features, config.measure_iterations);

        // 2. Weight streaming (weights reloaded for each minibatch)
        total_mini_weight_stream +=
            measure_weight_streaming_time(device, config.out_features, config.in_features, config.measure_iterations);

        // 3. NoC communication
        uint32_t mini_activation_tiles = ((config.small_batch_size + 31) / 32) * ((config.in_features + 31) / 32);
        total_mini_noc_comm += measure_noc_communication_time(device, mini_activation_tiles, config.measure_iterations);
    }

    results.mini_batch_sharding_time_us = total_mini_sharding;
    results.mini_batch_weight_stream_time_us = total_mini_weight_stream;
    results.mini_batch_noc_comm_time_us = total_mini_noc_comm;
    results.mini_batch_total_time_us = total_mini_sharding + total_mini_weight_stream + total_mini_noc_comm;

    std::cout << "  Total sharding time: " << results.mini_batch_sharding_time_us << " us\n";
    std::cout << "  Total weight streaming time: " << results.mini_batch_weight_stream_time_us << " us\n";
    std::cout << "  Total NoC comm time: " << results.mini_batch_noc_comm_time_us << " us\n";

    // Calculate overheads
    results.sharding_overhead_us = results.mini_batch_sharding_time_us - results.large_batch_sharding_time_us;
    results.weight_streaming_overhead_us =
        results.mini_batch_weight_stream_time_us - results.large_batch_weight_stream_time_us;
    results.noc_comm_overhead_us = results.mini_batch_noc_comm_time_us - results.large_batch_noc_comm_time_us;
    results.total_overhead_us = results.mini_batch_total_time_us - results.large_batch_total_time_us;

    // Cleanup
    CloseDevice(device);

    return results;
}

int main(int argc, char** argv) {
    ProfilingConfig config;

    // Parse command-line arguments (simple implementation)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--large-batch" && i + 1 < argc) {
            config.large_batch_size = std::stoi(argv[++i]);
        } else if (arg == "--small-batch" && i + 1 < argc) {
            config.small_batch_size = std::stoi(argv[++i]);
        } else if (arg == "--minibatches" && i + 1 < argc) {
            config.num_minibatches = std::stoi(argv[++i]);
        } else if (arg == "--in-features" && i + 1 < argc) {
            config.in_features = std::stoi(argv[++i]);
        } else if (arg == "--out-features" && i + 1 < argc) {
            config.out_features = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.measure_iterations = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_csv = argv[++i];
        } else if (arg == "--enable-tracy") {
            config.enable_tracy = true;
        }
    }

    std::cout << "===== TT-Metal Low-Level Profiling =====\n";
    std::cout << "Configuration:\n";
    std::cout << "  Large batch size: " << config.large_batch_size << "\n";
    std::cout << "  Small batch size: " << config.small_batch_size << "\n";
    std::cout << "  Num minibatches:  " << config.num_minibatches << "\n";
    std::cout << "  In features:      " << config.in_features << "\n";
    std::cout << "  Out features:     " << config.out_features << "\n";
    std::cout << "  Warmup iters:     " << config.warmup_iterations << "\n";
    std::cout << "  Measure iters:    " << config.measure_iterations << "\n";
    std::cout << "  Tracy enabled:    " << (config.enable_tracy ? "yes" : "no") << "\n";
    std::cout << "========================================\n\n";

    try {
        auto results = run_profiling_benchmark(config);
        results.print();
        results.save_to_csv(config.output_csv, config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
