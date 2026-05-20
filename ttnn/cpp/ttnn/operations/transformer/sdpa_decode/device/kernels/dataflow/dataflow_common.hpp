// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include <vector>
/******************************************************************************
 *                                                                             *
 *                   Common Functions for Dataflow Kernels                     *
 *                                                                             *
 ******************************************************************************/

/******************************************************************************
 *                   Generic Utility Functions                                 *
 ******************************************************************************/
template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

/******************************************************************************
 *                   Page Cache Functions            *
 ******************************************************************************/
template <typename PageT, uint32_t num_heads, uint32_t block_size_t, uint32_t Wt>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, const volatile tt_l1_ptr PageT* const page_table_ptr) {
    // Given some index in the sequence tiles in range [0, max_seq_len_t]
    // Return the physical tile id for that tile row
    constexpr uint32_t block_stride = num_heads * block_size_t * Wt;
    const uint32_t head_offset = cur_head * block_size_t * Wt;

    const uint32_t virtual_block = seq_tile_idx / block_size_t;

    const uint32_t physical_block = static_cast<uint32_t>(page_table_ptr[virtual_block]);
    const uint32_t block_row_offset = seq_tile_idx % block_size_t;
    const uint32_t block_offset = block_row_offset * Wt;
    return physical_block * block_stride + head_offset + block_offset;
}

// Backward-compatible overload (defaults to uint32_t page table entries)
template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, const volatile tt_l1_ptr uint32_t* const page_table_ptr) {
    return virtual_seq_tile_id_to_physical_tile_id<uint32_t, num_heads, block_size_t, Wt>(
        seq_tile_idx, cur_head, page_table_ptr);
}

/******************************************************************************
 *                   Generic Tile Manipulation Functions                       *
 ******************************************************************************/
template <uint32_t tile_bytes>
void copy_tile(uint64_t noc_read_addr_base, uint32_t q_write_ptr_base, uint32_t src_tile_id, uint32_t dst_tile_id) {
    noc_async_read(
        noc_read_addr_base + src_tile_id * tile_bytes, q_write_ptr_base + dst_tile_id * tile_bytes, tile_bytes);
}

template <uint32_t tile_bytes>
void fill_tile(uint32_t cb_id, uint32_t tile_id, uint32_t val) {
    if (val == 0) {
        constexpr uint32_t num_zeros_reads = tile_bytes / MEM_ZEROS_SIZE;
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        uint32_t write_addr = get_write_ptr(cb_id) + tile_id * tile_bytes;
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

        // Fill tile with zeros
        for (uint32_t i = 0; i < num_zeros_reads; ++i) {
            noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
            write_addr += MEM_ZEROS_SIZE;
        }
        noc_async_read_barrier();
    } else {
        // Fill 2 uint16 datums in each writes to optimize for performance
        volatile tt_l1_ptr uint32_t* ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
        constexpr int num_uint32_datums_tile = tile_bytes / 4;
        for (int k = 0; k < num_uint32_datums_tile; k++) {
            ptr[k] = val;
        }
    }
}

template <uint32_t tile_bytes>
void fill_tile_partial(uint32_t cb_id, uint32_t tile_id, uint32_t cur_pos_in_tile, uint32_t partial_val) {
    /*
    We want to fill cur_pos_in_tile + 1 to the end
    */
    constexpr int num_faces = (tile_bytes == 1024) ? 2 : 4;

    fill_tile<tile_bytes>(cb_id, tile_id, 0);
    if (cur_pos_in_tile == 31 || partial_val == 0) {
        return;
    }
    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    int face_start = (cur_pos_in_tile < 15) ? 0 : 1;
    uint32_t fill_pos_in_face = (cur_pos_in_tile + 1) % 16;
    if (face_start == 0) {
        // Fill 2 datums in each writes to optimize for performance
        constexpr int num_uint32_datums_tile_face = (16 * 16) / 2;
        for (int k = 1; k < num_faces; k += 2) {
            uint32_t uint32_face_idx = k << 7;
            for (int j = 0; j < num_uint32_datums_tile_face; j++) {
                uint32_ptr[uint32_face_idx + j] = partial_val;
            }
        }
    }

    // Again, optimizing performance by filling 2 uint16 datums in each write.
    // If the fill_pos_in_face is odd then we fill that pos with single datum,
    // otherwise we fill 2 datums in each write
    bool is_odd_pos_filled = fill_pos_in_face % 2 == 1;
    uint32_t fill_pos_in_uint32_face = (fill_pos_in_face + 1) >> 1;
    constexpr uint32_t num_cols_in_face = 16;
    constexpr uint32_t num_rows_in_face = 16;
    constexpr uint32_t num_cols_in_uint32_face = num_cols_in_face >> 1;
    for (int k = face_start; k < num_faces; k += 2) {
        uint32_t uint16_face_idx = k << 8;
        uint32_t uint32_face_idx = k << 7;

        for (uint32_t face_row_idx = 0; face_row_idx < num_rows_in_face; face_row_idx++) {
            // Here, if the fill_pos_in_face is odd then we fill that pos with single uint16 value
            if (is_odd_pos_filled) {
                uint16_ptr[uint16_face_idx + (fill_pos_in_face + num_cols_in_face * face_row_idx)] = datum_val;
            }

            for (uint32_t uint32_face_col_idx = fill_pos_in_uint32_face; uint32_face_col_idx < num_cols_in_uint32_face;
                 uint32_face_col_idx++) {
                uint32_ptr[uint32_face_idx + (uint32_face_col_idx + num_cols_in_uint32_face * face_row_idx)] =
                    partial_val;
            }
        }
    }
}

template <uint32_t tile_bytes>
void fill_tile_partial_sliding_window(uint32_t cb_id, uint32_t tile_id, uint32_t window_start_pos_in_tile, uint32_t partial_val) {
    /*
    For sliding window mask: fill positions 0 to window_start_pos_in_tile - 1 with partial_val (-inf)
    This is the inverse of fill_tile_partial which fills from cur_pos_in_tile + 1 to end

    Example: if window_start_pos_in_tile = 5, then positions 0,1,2,3,4 are filled with -inf
             and positions 5,6,7,...,31 remain as 0 (allowed)
    */
    constexpr int num_faces = (tile_bytes == 1024) ? 2 : 4;

    fill_tile<tile_bytes>(cb_id, tile_id, 0);
    if (window_start_pos_in_tile == 0 || partial_val == 0) {
        return;  // No masking needed if window starts at position 0 or no mask value
    }

    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);

    // Determine which faces to fill completely (before the window_start_pos_in_tile)
    int face_start = (window_start_pos_in_tile < 15) ? 0 : 1;  // Last face to fill completely

    // Fill complete faces (faces 0, 2, 4, 6... for faces before face_start)
    if (face_start == 1) {
        constexpr int num_uint32_datums_tile_face = (16 * 16) / 2;
        for (int k = 0; k < num_faces; k += 2) {
            uint32_t uint32_face_idx = k << 7;
            for (int j = 0; j < num_uint32_datums_tile_face; j++) {
                uint32_ptr[uint32_face_idx + j] = partial_val;
            }
        }
    }

    // Fill partial face (the face containing window_start_pos_in_tile)
    uint32_t fill_end_pos_in_face = window_start_pos_in_tile % 16;  // Position to stop filling (exclusive)

    // Optimize performance by filling 2 uint16 datums in each write
    bool is_odd_end_pos = fill_end_pos_in_face % 2 == 1;
    uint32_t fill_end_pos_in_uint32_face = fill_end_pos_in_face >> 1;
    constexpr uint32_t num_cols_in_face = 16;
    constexpr uint32_t num_rows_in_face = 16;
    constexpr uint32_t num_cols_in_uint32_face = num_cols_in_face >> 1;

    // Fill the face containing window_start_pos_in_tile
    int target_face = (window_start_pos_in_tile < 16) ? 0 : 1;
    for (int k = target_face; k < num_faces; k += 2) {
        uint32_t uint16_face_idx = k << 8;
        uint32_t uint32_face_idx = k << 7;

        for (uint32_t face_row_idx = 0; face_row_idx < num_rows_in_face; face_row_idx++) {
            // Fill uint32 pairs from start to fill_end_pos_in_uint32_face
            for (uint32_t uint32_face_col_idx = 0; uint32_face_col_idx < fill_end_pos_in_uint32_face; uint32_face_col_idx++) {
                uint32_ptr[uint32_face_idx + (uint32_face_col_idx + num_cols_in_uint32_face * face_row_idx)] = partial_val;
            }

            // Handle the odd position if fill_end_pos_in_face is odd
            if (is_odd_end_pos && fill_end_pos_in_face > 0) {
                uint16_ptr[uint16_face_idx + ((fill_end_pos_in_face - 1) + num_cols_in_face * face_row_idx)] = datum_val;
            }
        }
    }
}

/******************************************************************************
 *                   Attention Mask Functions                                 *
 ******************************************************************************/
template <
    uint32_t cb_mask_in,
    uint32_t mask_tile_bytes,
    uint32_t barrier_threshold,
    uint32_t PNHt,
    typename MaskReaderType>
uint32_t read_mask_chunk(
    uint32_t PSt,
    uint32_t Sk_chunk_t,
    uint32_t mask_chunk_tiles,
    uint32_t mask_start_tile_id,
    const MaskReaderType& mask_reader) {
    // Read mask chunk
    cb_reserve_back(cb_mask_in, mask_chunk_tiles);
    uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < PNHt; ++row) {
        uint32_t mask_tile_id = mask_start_tile_id + row * PSt;
        for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
            noc_async_read_tile(mask_tile_id, mask_reader, mask_write_ptr);
            mask_tile_id++;
            mask_write_ptr += mask_tile_bytes;

            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_mask_in, mask_chunk_tiles);
    mask_start_tile_id += mask_chunk_tiles;
    return mask_start_tile_id;
}

template <uint32_t cb_mask_in, uint32_t PNHt>
void generate_mask(uint32_t k_num_chunks, uint32_t Sk_chunk_t, uint32_t cur_pos) {
    /*
    example 1: 64 seqlen at cur_pos 40, 2 cores, 32 chunk size
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 40
    cur_pos_in_chunk = 8
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 8

    example 2: 1024 seqlen at cur_pos 990, 2 cores, 128 chunk size
    k_num_chunks = 8
    Sk_chunk_t = 4
    cur_pos = 990
    cur_pos_in_chunk = 94
    cur_pos_in_chunk_t = 2
    cur_pos_in_tile = 30

    example 3: 64 seqlen at cur_pos 63, 2 cores, 32 chunk size
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 63
    cur_pos_in_chunk = 31
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 31

    example 3: 64 seqlen at cur_pos 0, 2 cores, 32 chunk size
    k_num_chunks = 2
    Sk_chunk_t = 1
    cur_pos = 0
    cur_pos_in_chunk = 0
    cur_pos_in_chunk_t = 0
    cur_pos_in_tile = 0
    */

    // the cb_mask in is of size PNHt * Sk_chunk_t
    uint32_t total_read_tiles = PNHt * Sk_chunk_t;
    uint32_t cur_pos_in_chunk = cur_pos % (Sk_chunk_t * 32);
    uint32_t cur_pos_in_chunk_t = cur_pos_in_chunk / 32;
    uint32_t cur_pos_in_tile = cur_pos_in_chunk % 32;
    constexpr uint32_t NEG_INF = 0xFF80FF80;  // TODO: Make sure this is -inf

    cb_reserve_back(cb_mask_in, total_read_tiles);

    uint64_t noc_read_addr_base = get_noc_addr(get_read_ptr(cb_mask_in));
    uint32_t q_write_ptr_base = get_read_ptr(cb_mask_in);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        if (i < cur_pos_in_chunk_t) {
            // fill with zero
            if (i == 0) {
                fill_tile<tile_bytes>(cb_mask_in, i, 0);
            } else {
                copy_tile<tile_bytes>(
                    noc_read_addr_base, q_write_ptr_base, 0, i);  // copy from cb_mask_in[0] to cb_mask_in[i]
                if (i == cur_pos_in_chunk_t - 1) {
                    noc_async_read_barrier();
                }
            }
        } else if (i == cur_pos_in_chunk_t) {
            // fill with partial zero/-inf
            fill_tile_partial<tile_bytes>(cb_mask_in, i, cur_pos_in_tile, NEG_INF);
        } else {
            // fill with -inf
            if (i == cur_pos_in_chunk_t + 1) {
                fill_tile<tile_bytes>(cb_mask_in, i, NEG_INF);
            } else {
                copy_tile<tile_bytes>(
                    noc_read_addr_base,
                    q_write_ptr_base,
                    cur_pos_in_chunk_t + 1,
                    i);  // copy from cb_mask_in[cur_pos_in_chunk_t+1] to cb_mask_in[i]
                if (i == Sk_chunk_t - 1) {
                    noc_async_read_barrier();
                }
            }
        }
        for (uint32_t j = 1; j < PNHt; ++j) {
            // copy from cb_mask_in[i] to cb_mask_in[j*Sk_chunk_t + i]
            copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, i, j * Sk_chunk_t + i);
            if (j == PNHt - 1) {
                noc_async_read_barrier();
            }
        }
    }

    cb_push_back(cb_mask_in, total_read_tiles);
}

template <uint32_t cb_mask_in, uint32_t PNHt>
void generate_sliding_window_mask(uint32_t k_num_chunks, uint32_t Sk_chunk_t, uint32_t window_start) {
    /*
    Generate sliding window mask for the first chunk:
    - Mask positions < window_start with -inf (sliding window start)
    - Allow positions >= window_start

    This mask is applied only to the first chunk to enforce sliding window constraint.
    */

    // the cb_mask in is of size PNHt * Sk_chunk_t
    uint32_t total_read_tiles = PNHt * Sk_chunk_t;
    uint32_t window_start_in_chunk = window_start % (Sk_chunk_t * 32);
    uint32_t window_start_in_chunk_t = window_start_in_chunk / 32;
    uint32_t window_start_in_tile = window_start_in_chunk % 32;
    constexpr uint32_t NEG_INF = 0xFF80FF80;  // TODO: Make sure this is -inf

    cb_reserve_back(cb_mask_in, total_read_tiles);

    uint64_t noc_read_addr_base = get_noc_addr(get_read_ptr(cb_mask_in));
    uint32_t q_write_ptr_base = get_read_ptr(cb_mask_in);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        if (i < window_start_in_chunk_t) {
            // Tile is completely before sliding window - fill with -inf
            if (i == 0) {
                fill_tile<tile_bytes>(cb_mask_in, i, NEG_INF);
            } else {
                copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, 0, i);
            }
        } else if (i == window_start_in_chunk_t) {
            // Tile contains sliding window start - partial mask at beginning
            fill_tile_partial_sliding_window<tile_bytes>(cb_mask_in, i, window_start_in_tile, NEG_INF);
        } else {
            // Tile is within sliding window - fill with zeros (allow)
            if (i == window_start_in_chunk_t + 1) {
                fill_tile<tile_bytes>(cb_mask_in, i, 0);
            } else {
                // Copy from the first allowed tile
                copy_tile<tile_bytes>(
                    noc_read_addr_base,
                    q_write_ptr_base,
                    window_start_in_chunk_t + 1,
                    i);  // copy from cb_mask_in[cur_pos_in_chunk_t+1] to cb_mask_in[i]
                if (i == Sk_chunk_t - 1) {
                    noc_async_read_barrier();
                }
            }
        }

        // Copy to all heads
        for (uint32_t j = 1; j < PNHt; ++j) {
            copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, i, j * Sk_chunk_t + i);
            if (j == PNHt - 1) {
                noc_async_read_barrier();
            }
        }
    }

    cb_push_back(cb_mask_in, total_read_tiles);
}

/******************************************************************************
 *                   Writer Kernel Specific Functions                         *
 ******************************************************************************/

template <
    uint32_t out_chunk_tiles,
    uint32_t cb_out,
    uint32_t cb_out_m,
    uint32_t cb_out_l,
    uint32_t cb_intermed_out,
    uint32_t PNHt>
void worker_compute(
    uint64_t in0_sender_semaphore_noc_addr,
    uint32_t worker_id,
    uint32_t reduce_core_noc_x,
    uint32_t reduce_core_noc_y) {
    uint32_t out_tile_id = 0;

    // Wait for compute to deliver output chunk
    cb_wait_front(cb_out, out_chunk_tiles);
    cb_wait_front(cb_out_m, PNHt);
    cb_wait_front(cb_out_l, PNHt);

    // Write output chunk to reducer
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    uint32_t worker_offset = worker_id * (out_chunk_tiles + 2 * PNHt) * tile_bytes;
    constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes;
    constexpr uint32_t ml_write_size = PNHt * tile_bytes;
    uint64_t output_write_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, get_write_ptr(cb_intermed_out)) + worker_offset;
    noc_async_write(get_read_ptr(cb_out), output_write_addr, o_write_size);
    output_write_addr += o_write_size;
    noc_async_write(get_read_ptr(cb_out_m), output_write_addr, ml_write_size);
    output_write_addr += ml_write_size;
    noc_async_write(get_read_ptr(cb_out_l), output_write_addr, ml_write_size);

    // increment semaphore
    noc_async_write_barrier();
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);

    // pop front
    cb_pop_front(cb_out, out_chunk_tiles);
    cb_pop_front(cb_out_m, PNHt);
    cb_pop_front(cb_out_l, PNHt);
}

template <uint32_t cb_out, uint32_t out_chunk_tiles, uint32_t barrier_threshold, typename WriterType>
uint32_t write_tiles_to_memory(uint32_t& out_tile_id, const WriterType& out_writer, uint32_t& barrier_count) {
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
        noc_async_write_tile(out_tile_id, out_writer, l1_read_addr);
        ++out_tile_id;
        l1_read_addr += tile_bytes;
        if (++barrier_count == barrier_threshold) {
            noc_async_writes_flushed();
            barrier_count = 0;
        }
    }
    return barrier_count;
}

template <uint32_t cb_out, uint32_t ELEMENT_SIZE, uint32_t barrier_threshold, typename WriterType>
uint32_t write_partial_tiles_to_memory(
    uint32_t& out_tile_id,
    const WriterType& out_writer,
    uint32_t& barrier_count,
    uint32_t cur_head,
    uint32_t num_heads_to_write,
    uint32_t out_chunk_tiles) {
    constexpr uint32_t FACE_HW = 16;
    constexpr uint32_t FACE_ELEMENT_CNT = FACE_HW * FACE_HW;  // 256
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t FACE_LINE_BYTES = FACE_HW * ELEMENT_SIZE;

    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
        uint64_t out_writer_noc_addr = get_noc_addr(out_tile_id, out_writer);
        uint32_t l1_read_addr = get_read_ptr(cb_out) + tile * tile_bytes;

        // write partial output for each head
        for (uint32_t head = 0; head < num_heads_to_write; ++head) {
            uint32_t starting_row = cur_head * num_heads_to_write + head;
            uint32_t in_tile_offset_by_starting_head =
                starting_row < FACE_HW
                    ? starting_row * FACE_LINE_BYTES
                    : (starting_row + FACE_HW) * FACE_LINE_BYTES;  // Skip the second face which has FACE_HW rows
            uint64_t out_writer_noc_addr_head = out_writer_noc_addr + in_tile_offset_by_starting_head;
            uint32_t l1_read_addr_head = l1_read_addr + in_tile_offset_by_starting_head;

            // Write first phase
            noc_async_write(l1_read_addr_head, out_writer_noc_addr_head, FACE_LINE_BYTES);

            // Write second phase
            noc_async_write(
                l1_read_addr_head + FACE_ELEMENT_CNT * ELEMENT_SIZE,
                out_writer_noc_addr_head + FACE_ELEMENT_CNT * ELEMENT_SIZE,
                FACE_LINE_BYTES);

            if (++barrier_count == barrier_threshold) {
                noc_async_writes_flushed();
                barrier_count = 0;
            }
        }

        ++out_tile_id;
    }
    return barrier_count;
}

/******************************************************************************
 *                   Reader Kernel Specific Functions                         *
 ******************************************************************************/

template <
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t barrier_threshold,
    uint32_t mask_tile_bytes,
    uint32_t PNHt,
    bool use_attention_mask,
    uint32_t cb_k_in,
    uint32_t cb_v_in,
    uint32_t cb_mask_in,
    bool reuse_k,  // If enabled, read V from K, instead of from DRAM
    typename KReaderType,
    typename VReaderType,
    typename MaskReaderType>
void read_kv_mask_chunks(
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    uint32_t k_start_tile_id,
    uint32_t mask_start_tile_id,
    uint32_t Sk_chunk_t,
    uint32_t k_chunk_tiles,
    uint32_t v_chunk_tiles,
    uint32_t mask_chunk_tiles,
    const KReaderType& k_reader,
    const VReaderType& v_reader,
    const MaskReaderType& mask_reader,
    uint32_t k_tile_bytes,
    uint32_t v_tile_bytes,
    uint32_t PSt) {
    uint32_t barrier_count = 0;
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        // Read K chunk transposed
        cb_reserve_back(cb_k_in, k_chunk_tiles);
        uint32_t k_write_ptr = get_write_ptr(cb_k_in);
        uint64_t k_base_read_ptr = get_noc_addr(k_write_ptr);
        barrier_count = 0;
        for (uint32_t col = 0; col < DHt; ++col) {
            uint32_t k_tile_id = k_start_tile_id + col;
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);
                if (++barrier_count == barrier_threshold) {
                    noc_async_read_barrier();
                    barrier_count = 0;
                }
                k_tile_id += DHt;
                k_write_ptr += k_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, k_chunk_tiles);

        if constexpr (use_attention_mask) {
            mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
                PSt, Sk_chunk_t, mask_chunk_tiles, mask_start_tile_id, mask_reader);
        }

        // Read V chunk (tranpose of K), from K's L1 buffer
        if constexpr (reuse_k) {
            cb_reserve_back(cb_v_in, v_chunk_tiles);
            uint32_t v_write_ptr = get_write_ptr(cb_v_in);
            uint64_t k_read_ptr = k_base_read_ptr;
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {       // Row of V
                k_read_ptr = k_base_read_ptr + row * k_tile_bytes;  // Increment across K's Col

                for (uint32_t col = 0; col < vDHt; ++col) {  // Col of V
                    noc_async_read(k_read_ptr, v_write_ptr, v_tile_bytes);

                    v_write_ptr += v_tile_bytes;
                    k_read_ptr += Sk_chunk_t * k_tile_bytes;  // Strid across K's width
                }
            }
        } else {
            cb_reserve_back(cb_v_in, v_chunk_tiles);
            uint32_t v_write_ptr = get_write_ptr(cb_v_in);
            barrier_count = 0;
            uint32_t v_tile_id = k_start_tile_id;
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                for (uint32_t col = 0; col < vDHt; ++col) {
                    noc_async_read_tile(v_tile_id, v_reader, v_write_ptr);
                    if (++barrier_count == barrier_threshold) {
                        noc_async_read_barrier();
                        barrier_count = 0;
                    }
                    v_tile_id++;
                    v_write_ptr += v_tile_bytes;
                }
                v_tile_id += (DHt - vDHt);  // Skip the padding!
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_v_in, v_chunk_tiles);

        // Update the starting tile id for next iteration
        k_start_tile_id += k_chunk_tiles;
    }
}

// N-tier flat-range version: dispatches each tile to the tier whose token range contains it, or DRAM.
// tier_start_tiles[i] and tier_size_tiles[i] define the flat token range for tier i.
// head_base is (batch_offset + head_offset) in UNITS OF ONE HEAD's stride within the tier tensor.
// Tile ids within each tier tensor: head_base * tier_size_tiles[i] * DHt + local_row * DHt + col
template <
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t barrier_threshold,
    uint32_t mask_tile_bytes,
    uint32_t PNHt,
    bool use_attention_mask,
    uint32_t cb_k_in,
    uint32_t cb_v_in,
    uint32_t cb_mask_in,
    bool reuse_k,
    uint32_t num_tiers,
    typename KReaderType,
    typename VReaderType,
    typename MaskReaderType,
    typename L1K0,
    typename L1V0,
    typename L1K1,
    typename L1V1,
    typename L1K2,
    typename L1V2,
    typename L1K3,
    typename L1V3,
    typename L1K4,
    typename L1V4>
void read_kv_mask_chunks_n_tier(
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    uint32_t k_start_tile_id,
    uint32_t mask_start_tile_id,
    uint32_t Sk_chunk_t,
    uint32_t k_chunk_tiles,
    uint32_t v_chunk_tiles,
    uint32_t mask_chunk_tiles,
    const KReaderType& k_reader,
    const VReaderType& v_reader,
    const MaskReaderType& mask_reader,
    uint32_t k_tile_bytes,
    uint32_t v_tile_bytes,
    uint32_t PSt,
    // N-tier L1 params
    uint32_t l1_kv_head_base,  // (batch_offset + head_offset) in head units
    uint32_t l1_min_expected_hit_ratio_mille,
    const uint32_t* tier_start_tiles,  // [5] flat L1 tile-offset where each tier starts
    const uint32_t* tier_size_tiles,   // [5] tier size in tile units
    // Ring-buffer semantics (Option A fix):
    //   The Python write path stores token K/V at L1[cur_pos % T_total], where
    //   T_total = sum(tier_size_tiles[0..num_tiers)) * TILE_HEIGHT. Without ring-aware
    //   reads, the kernel naively returned L1[gst] for sequence tile gst — which
    //   reads stale data once cur_pos >= T_total (the slot has been overwritten by
    //   a later token). These two extra params let the kernel compute the
    //   fresh-token window [cur_pos - T_total + 1, cur_pos] and ring-remap reads.
    //   See research_codes/documents/l1_kv_cache_cache/per_core_validate_walkthrough.md
    //   for the bug description.
    uint32_t cur_pos_tokens,
    uint32_t total_l1_tiles,
    const L1K0& l1_k0_rd,
    const L1V0& l1_v0_rd,
    const L1K1& l1_k1_rd,
    const L1V1& l1_v1_rd,
    const L1K2& l1_k2_rd,
    const L1V2& l1_v2_rd,
    const L1K3& l1_k3_rd,
    const L1V3& l1_v3_rd,
    const L1K4& l1_k4_rd,
    const L1V4& l1_v4_rd) {
    // Compute the fresh-tile window on the sequence axis. Only tiles whose 32
    // tokens are ALL within the ring's currently-live range are read from L1;
    // tiles that straddle the ring boundary fall back to DRAM (an extra read
    // for at most one tile per chunk, but correctness > tier hit-rate).
    constexpr uint32_t TILE_HEIGHT_LOCAL = 32;
    const uint32_t total_l1_tokens = total_l1_tiles * TILE_HEIGHT_LOCAL;
    uint32_t fresh_lo_tile;
    uint32_t fresh_hi_tile;
    if (cur_pos_tokens + 1u <= total_l1_tokens) {
        // No ring wrap yet: fresh tile range is [0, floor((cur_pos+1)/TILE_HEIGHT)).
        fresh_lo_tile = 0u;
        fresh_hi_tile = (cur_pos_tokens + 1u) / TILE_HEIGHT_LOCAL;
    } else {
        // Wrap: fresh tokens are (cur_pos - T_total, cur_pos]; convert to fully-
        // contained tile range, ceiling on lo, floor on hi.
        uint32_t earliest_fresh_token = cur_pos_tokens + 1u - total_l1_tokens;
        fresh_lo_tile = (earliest_fresh_token + TILE_HEIGHT_LOCAL - 1u) / TILE_HEIGHT_LOCAL;
        fresh_hi_tile = (cur_pos_tokens + 1u) / TILE_HEIGHT_LOCAL;
    }

    // Hit-ratio heuristic: count fresh tiles vs total tiles to read.
    uint32_t seq_tiles = k_chunk_end * Sk_chunk_t;
    uint32_t hot_tiles = (fresh_hi_tile > fresh_lo_tile) ? (fresh_hi_tile - fresh_lo_tile) : 0u;
    uint32_t expected_hit_ratio_mille = seq_tiles == 0 ? 0 : (hot_tiles * 1000) / seq_tiles;
    bool enable_l1_reads =
        hot_tiles > 0 && total_l1_tiles > 0 && expected_hit_ratio_mille >= l1_min_expected_hit_ratio_mille;

    uint32_t barrier_count = 0;
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        uint32_t chunk_seq_tile = k_chunk * Sk_chunk_t;

        cb_reserve_back(cb_k_in, k_chunk_tiles);
        uint32_t k_write_ptr = get_write_ptr(cb_k_in);
        uint64_t k_base_read_ptr = get_noc_addr(k_write_ptr);
        barrier_count = 0;

        // Helper lambda: given global sequence tile gst, return (tier_idx, flat_tile)
        // for an L1 read, or {0xFFFFFFFF, _} to mean DRAM. The fresh-window check
        // gates L1 entirely for stale or future tiles; flat_tile is gst mapped through
        // the ring (gst % total_l1_tiles) and then dispatched to whichever tier owns
        // it in the flat L1 layout.
        auto find_tier = [&](uint32_t gst, uint32_t& out_flat_tile) -> uint32_t {
            out_flat_tile = 0u;
            if (!enable_l1_reads) {
                return 0xFFFFFFFFu;
            }
            if (gst < fresh_lo_tile || gst >= fresh_hi_tile) {
                return 0xFFFFFFFFu;
            }
            uint32_t flat_tile = (total_l1_tiles == 0) ? gst : (gst % total_l1_tiles);
            out_flat_tile = flat_tile;
            if constexpr (num_tiers >= 1) {
                if (flat_tile >= tier_start_tiles[0] && flat_tile < tier_start_tiles[0] + tier_size_tiles[0]) {
                    return 0u;
                }
            }
            if constexpr (num_tiers >= 2) {
                if (flat_tile >= tier_start_tiles[1] && flat_tile < tier_start_tiles[1] + tier_size_tiles[1]) {
                    return 1u;
                }
            }
            if constexpr (num_tiers >= 3) {
                if (flat_tile >= tier_start_tiles[2] && flat_tile < tier_start_tiles[2] + tier_size_tiles[2]) {
                    return 2u;
                }
            }
            if constexpr (num_tiers >= 4) {
                if (flat_tile >= tier_start_tiles[3] && flat_tile < tier_start_tiles[3] + tier_size_tiles[3]) {
                    return 3u;
                }
            }
            if constexpr (num_tiers >= 5) {
                if (flat_tile >= tier_start_tiles[4] && flat_tile < tier_start_tiles[4] + tier_size_tiles[4]) {
                    return 4u;
                }
            }
            return 0xFFFFFFFFu;
        };

        // Read K chunk (transposed: col-major outer loop)
        for (uint32_t col = 0; col < DHt; ++col) {
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                uint32_t gst = chunk_seq_tile + row;
                uint32_t flat_tile = 0;
                uint32_t tier_idx = find_tier(gst, flat_tile);
                if (tier_idx != 0xFFFFFFFFu) {
                    uint32_t local_row = flat_tile - tier_start_tiles[tier_idx];
                    uint32_t l1_k_tile_id = l1_kv_head_base * tier_size_tiles[tier_idx] * DHt + local_row * DHt + col;
                    if constexpr (num_tiers >= 1) {
                        if (tier_idx == 0u) {
                            noc_async_read_tile(l1_k_tile_id, l1_k0_rd, k_write_ptr);
                            goto k_done;
                        }
                    }
                    if constexpr (num_tiers >= 2) {
                        if (tier_idx == 1u) {
                            noc_async_read_tile(l1_k_tile_id, l1_k1_rd, k_write_ptr);
                            goto k_done;
                        }
                    }
                    if constexpr (num_tiers >= 3) {
                        if (tier_idx == 2u) {
                            noc_async_read_tile(l1_k_tile_id, l1_k2_rd, k_write_ptr);
                            goto k_done;
                        }
                    }
                    if constexpr (num_tiers >= 4) {
                        if (tier_idx == 3u) {
                            noc_async_read_tile(l1_k_tile_id, l1_k3_rd, k_write_ptr);
                            goto k_done;
                        }
                    }
                    if constexpr (num_tiers >= 5) {
                        if (tier_idx == 4u) {
                            noc_async_read_tile(l1_k_tile_id, l1_k4_rd, k_write_ptr);
                            goto k_done;
                        }
                    }
                k_done:;
                } else {
                    noc_async_read_tile(k_start_tile_id + col + row * DHt, k_reader, k_write_ptr);
                }
                k_write_ptr += k_tile_bytes;
                if (++barrier_count == barrier_threshold) {
                    noc_async_read_barrier();
                    barrier_count = 0;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, k_chunk_tiles);

        if constexpr (use_attention_mask) {
            mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
                PSt, Sk_chunk_t, mask_chunk_tiles, mask_start_tile_id, mask_reader);
        }

        // Read V chunk (row-major: sequence row outer loop)
        cb_reserve_back(cb_v_in, v_chunk_tiles);
        uint32_t v_write_ptr = get_write_ptr(cb_v_in);
        barrier_count = 0;
        for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
            uint32_t gst = chunk_seq_tile + row;
            uint32_t flat_tile = 0;
            uint32_t tier_idx = find_tier(gst, flat_tile);
            if (tier_idx != 0xFFFFFFFFu) {
                uint32_t local_row = flat_tile - tier_start_tiles[tier_idx];
                uint32_t l1_v_tile_id_base = l1_kv_head_base * tier_size_tiles[tier_idx] * vDHt + local_row * vDHt;
                for (uint32_t col = 0; col < vDHt; ++col) {
                    uint32_t l1_v_tile_id = l1_v_tile_id_base + col;
                    if constexpr (num_tiers >= 1) {
                        if (tier_idx == 0u) {
                            noc_async_read_tile(l1_v_tile_id, l1_v0_rd, v_write_ptr);
                            goto v_done;
                        }
                    }
                    if constexpr (num_tiers >= 2) {
                        if (tier_idx == 1u) {
                            noc_async_read_tile(l1_v_tile_id, l1_v1_rd, v_write_ptr);
                            goto v_done;
                        }
                    }
                    if constexpr (num_tiers >= 3) {
                        if (tier_idx == 2u) {
                            noc_async_read_tile(l1_v_tile_id, l1_v2_rd, v_write_ptr);
                            goto v_done;
                        }
                    }
                    if constexpr (num_tiers >= 4) {
                        if (tier_idx == 3u) {
                            noc_async_read_tile(l1_v_tile_id, l1_v3_rd, v_write_ptr);
                            goto v_done;
                        }
                    }
                    if constexpr (num_tiers >= 5) {
                        if (tier_idx == 4u) {
                            noc_async_read_tile(l1_v_tile_id, l1_v4_rd, v_write_ptr);
                            goto v_done;
                        }
                    }
                v_done:;
                    v_write_ptr += v_tile_bytes;
                    if (++barrier_count == barrier_threshold) {
                        noc_async_read_barrier();
                        barrier_count = 0;
                    }
                }
            } else {
                uint32_t dram_v_tile_id = k_start_tile_id + row * DHt;
                for (uint32_t col = 0; col < vDHt; ++col) {
                    noc_async_read_tile(dram_v_tile_id + col, v_reader, v_write_ptr);
                    v_write_ptr += v_tile_bytes;
                    if (++barrier_count == barrier_threshold) {
                        noc_async_read_barrier();
                        barrier_count = 0;
                    }
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_v_in, v_chunk_tiles);

        k_start_tile_id += k_chunk_tiles;
    }
}

template <
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t barrier_threshold,
    uint32_t mask_tile_bytes,
    uint32_t PNHt,
    bool use_attention_mask,
    uint32_t cb_k_in,
    uint32_t cb_v_in,
    uint32_t cb_mask_in,
    bool reuse_k,
    typename KReaderType,
    typename VReaderType,
    typename MaskReaderType,
    typename L1KReaderType,
    typename L1VReaderType>
void read_kv_mask_chunks_dual_source(
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    uint32_t k_start_tile_id,
    uint32_t mask_start_tile_id,
    uint32_t Sk_chunk_t,
    uint32_t k_chunk_tiles,
    uint32_t v_chunk_tiles,
    uint32_t mask_chunk_tiles,
    const KReaderType& k_reader,
    const VReaderType& v_reader,
    const MaskReaderType& mask_reader,
    uint32_t k_tile_bytes,
    uint32_t v_tile_bytes,
    uint32_t PSt,
    // L1 dual-source params
    const L1KReaderType& l1_k_reader,
    const L1VReaderType& l1_v_reader,
    uint32_t l1_recent_window_start_tile,
    uint32_t l1_recent_window_size_tiles,
    uint32_t l1_sink_size_tiles,
    uint32_t l1_min_expected_hit_ratio_mille,
    uint32_t l1_k_start_tile_id_for_head,
    uint32_t l1_v_start_tile_id_for_head) {
    uint32_t barrier_count = 0;
    uint32_t seq_tiles = k_chunk_end * Sk_chunk_t;
    uint32_t hot_tiles = seq_tiles < l1_sink_size_tiles ? seq_tiles : l1_sink_size_tiles;
    if (l1_recent_window_size_tiles > 0 && seq_tiles > l1_sink_size_tiles) {
        uint32_t recent_candidates = seq_tiles - l1_sink_size_tiles;
        hot_tiles += recent_candidates < l1_recent_window_size_tiles ? recent_candidates : l1_recent_window_size_tiles;
    }
    uint32_t expected_hit_ratio_mille = seq_tiles == 0 ? 0 : (hot_tiles * 1000) / seq_tiles;
    bool enable_l1_reads = (l1_sink_size_tiles > 0 || l1_recent_window_size_tiles > 0) &&
                           expected_hit_ratio_mille >= l1_min_expected_hit_ratio_mille;
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        // Determine the sequence tile row for this chunk
        uint32_t chunk_seq_tile = k_chunk * Sk_chunk_t;
#define DEBUG_PRINT 1
#if defined(DEBUG_PRINT)
        uint32_t k_l1_hits = 0;
        uint32_t k_dram_hits = 0;
        uint32_t v_l1_hits = 0;
        uint32_t v_dram_hits = 0;
#endif

        // We must check L1 inclusion at tile granularity, because a chunk may straddle the L1 boundary.
        cb_reserve_back(cb_k_in, k_chunk_tiles);
        uint32_t k_write_ptr = get_write_ptr(cb_k_in);
        uint64_t k_base_read_ptr = get_noc_addr(k_write_ptr);
        barrier_count = 0;

        for (uint32_t col = 0; col < DHt; ++col) {
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                uint32_t global_seq_tile = chunk_seq_tile + row;
                bool in_sink = enable_l1_reads && (global_seq_tile < l1_sink_size_tiles);
                bool in_recent = enable_l1_reads && (l1_recent_window_size_tiles > 0) &&
                                 (global_seq_tile >= l1_recent_window_start_tile) &&
                                 (global_seq_tile < l1_recent_window_start_tile + l1_recent_window_size_tiles);

                if (in_sink || in_recent) {
                    uint32_t l1_tile_row = in_sink ? global_seq_tile
                                                   : (l1_sink_size_tiles + ((global_seq_tile - l1_sink_size_tiles) %
                                                                            l1_recent_window_size_tiles));
                    uint32_t l1_k_tile_id = l1_k_start_tile_id_for_head + l1_tile_row * DHt + col;
                    noc_async_read_tile(l1_k_tile_id, l1_k_reader, k_write_ptr);
#if defined(DEBUG_PRINT)
                    k_l1_hits++;
#endif
                } else {
                    uint32_t dram_k_tile_id = k_start_tile_id + col + row * DHt;
                    noc_async_read_tile(dram_k_tile_id, k_reader, k_write_ptr);
#if defined(DEBUG_PRINT)
                    k_dram_hits++;
#endif
                }
                k_write_ptr += k_tile_bytes;  // Linear sequential write matching DRAM layout

                if (++barrier_count == barrier_threshold) {
                    noc_async_read_barrier();
                    barrier_count = 0;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, k_chunk_tiles);

        if constexpr (use_attention_mask) {
            mask_start_tile_id = read_mask_chunk<cb_mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
                PSt, Sk_chunk_t, mask_chunk_tiles, mask_start_tile_id, mask_reader);
        }

        // Read V chunk row by row
        cb_reserve_back(cb_v_in, v_chunk_tiles);
        uint32_t v_write_ptr = get_write_ptr(cb_v_in);
        barrier_count = 0;

        for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
            uint32_t global_seq_tile = chunk_seq_tile + row;
            bool in_sink = enable_l1_reads && (global_seq_tile < l1_sink_size_tiles);
            bool in_recent = enable_l1_reads && (l1_recent_window_size_tiles > 0) &&
                             (global_seq_tile >= l1_recent_window_start_tile) &&
                             (global_seq_tile < l1_recent_window_start_tile + l1_recent_window_size_tiles);

            uint32_t dram_v_tile_id = k_start_tile_id + row * DHt;
            uint32_t l1_v_tile_id = 0;
            if (in_sink || in_recent) {
                uint32_t l1_tile_row =
                    in_sink
                        ? global_seq_tile
                        : (l1_sink_size_tiles + ((global_seq_tile - l1_sink_size_tiles) % l1_recent_window_size_tiles));
                l1_v_tile_id = l1_v_start_tile_id_for_head + l1_tile_row * vDHt;
            }

            for (uint32_t col = 0; col < vDHt; ++col) {
                if (in_sink || in_recent) {
                    noc_async_read_tile(l1_v_tile_id + col, l1_v_reader, v_write_ptr);
#if defined(DEBUG_PRINT)
                    v_l1_hits++;
#endif
                } else {
                    noc_async_read_tile(dram_v_tile_id + col, v_reader, v_write_ptr);
#if defined(DEBUG_PRINT)
                    v_dram_hits++;
#endif
                }
                v_write_ptr += v_tile_bytes;
                if (++barrier_count == barrier_threshold) {
                    noc_async_read_barrier();
                    barrier_count = 0;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_v_in, v_chunk_tiles);

#if defined(DEBUG_PRINT)
        DPRINT << "CHUNK " << k_chunk << " | K(L1:" << k_l1_hits << " DRAM:" << k_dram_hits << ") | V(L1:" << v_l1_hits
               << " DRAM:" << v_dram_hits << ")" << ENDL();
#endif

        k_start_tile_id += k_chunk_tiles;
    }
}
