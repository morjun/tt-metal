// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * MODIFIED VERSION with detailed profiling zones for COMPUTE kernel
 *
 * This shows exactly what operations TRISC is performing and their timing.
 */

#include <cstdint>
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

// ✅ ADD THIS: Enable kernel profiling
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    // ✅ ADD THIS: Main profiling scope
    DeviceZoneScopedMainChildN("TRISC-MATMUL-COMPUTE");

    uint32_t in0_block_w = get_compile_time_arg_val(0);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);
    uint32_t num_blocks = get_compile_time_arg_val(7);
    uint32_t out_subblock_h = get_compile_time_arg_val(8);
    uint32_t out_subblock_w = get_compile_time_arg_val(9);
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);
    uint32_t batch = get_compile_time_arg_val(11);

    {
        // ✅ ADD THIS: Profile MM initialization
        DeviceZoneScopedN("MM-INIT");
        mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24);
    }

    for (uint32_t b = 0; b < batch; b++) {
        // ✅ ADD THIS: Profile each batch
        DeviceZoneScopedN("BATCH-ITERATION");

        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            // ✅ ADD THIS: Profile each block
            DeviceZoneScopedN("BLOCK-ITERATION");

            bool last_out = block == (num_blocks - 1);

            {
                // ✅ ADD THIS: Measure CB wait time (IDLE!)
                DeviceZoneScopedN("CB-WAIT-FRONT");
                cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
                cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);
            }

            // ✅ ADD THIS: New zone to isolate processing from synchronization
            {
                DeviceZoneScopedN("GEMM-PROCESSING");

                int in0_index_subblock_offset = 0;
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        {
                            // ✅ ADD THIS: Measure DST acquisition
                            DeviceZoneScopedN("ACQUIRE-DST");
                            acquire_dst();
                        }

                        if (enable_reload) {
                            // ✅ ADD THIS: Profile reload operation
                            DeviceZoneScopedN("RELOAD-PARTIAL");

                            copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_1, tt::CBIndex::c_24);
                            cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                copy_tile(tt::CBIndex::c_24, i, i);
                            }
                            cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                            mm_init_short_with_dt(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24);
                        }

                        {
                            // ✅ ADD THIS: THE ACTUAL MATMUL COMPUTATION
                            DeviceZoneScopedN("MATMUL-TILES");

                            // Compute output sub-block from in0_subblock x in1_subblock
                            int dst_index = 0;
                            int in0_index_h_offset = 0;
                            for (uint32_t h = 0; h < out_subblock_h; h++) {
                                for (uint32_t w = 0; w < out_subblock_w; w++) {
                                    int in1_index_inner_dim_offset = 0;
                                    for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                        int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                        int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                        matmul_tiles(
                                            tt::CBIndex::c_0,
                                            tt::CBIndex::c_1,
                                            in0_index,
                                            in1_index,
                                            dst_index,
                                            false /* transpose */);
                                        in1_index_inner_dim_offset += in1_per_core_w;
                                    }
                                    dst_index++;
                                }
                                in0_index_h_offset += in0_block_w;
                            }
                        }

                        {
                            // ✅ ADD THIS: Profile packing/output
                            DeviceZoneScopedN("PACK-OUTPUT");

                            if (last_out) {
                                // Pack out to output buffer
                                cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                    pack_tile(i, tt::CBIndex::c_16);
                                }
                                cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                            } else {
                                // Wait for tiles in output buffer to be written out
                                if (block == 0) {
                                    cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
                                    out_num_tiles_to_wait += out_subblock_num_tiles;
                                }
                                // Move partial result to interm buffer
                                cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                    pack_tile(i, tt::CBIndex::c_24);
                                }
                                cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                            }
                        }

                        {
                            // ✅ ADD THIS: Profile DST release
                            DeviceZoneScopedN("RELEASE-DST");
                            release_dst();
                        }

                        in1_index_subblock_offset += out_subblock_w;
                    }
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }
            }  // End of GEMM-PROCESSING

            {
                // ✅ ADD THIS: Profile CB pop
                DeviceZoneScopedN("CB-POP-FRONT");
                cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
                cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
            }

            enable_reload = spill;
        }
    }
}
}  // namespace NAMESPACE
