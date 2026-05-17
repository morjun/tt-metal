# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass
from typing import List

import torch
from loguru import logger

import ttnn
from models.common.llama_models import (
    CompletionMessage,
    StopReason,
    TokenResult,
    create_vision_mask,
    encode_content,
    extract_images_from_messages,
    sample_top_p,
)
from models.common.tt_sampling import format_sampling_params
from models.tt_transformers.tt import l1_kv_perf
from models.tt_transformers.tt.common import (
    copy_host_to_device,
    get_block_size,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.model_config import CheckpointType


@dataclass(frozen=True)
class SamplingParams:
    """
    Used in Generator decode forward functions for greedy decoding / sampling on device.
    The same data class exists in vLLM at vllm/worker/tt_model_runner.py.
    """

    temperature: float | list[float]
    top_k: int | list[int]
    top_p: float | list[float]


# Split lists into chunks
def split_list(lst, n):
    """Split list into n equal parts"""
    chunk_size = len(lst) // n
    chunks = []
    start = 0
    for i in range(n):
        chunks.append(list(lst[start : start + chunk_size]))  # Convert to list explicitly
        start = end
    return chunks


class Generator:
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        """
        Creating a LlamaVision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        LlamaVision is general to text and chat.

        For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.

        """
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.processor = processor
        self.tokenizer = tokenizer
        self.data_parallel = len(self.model)
        self.prev_page_table = None
        self.trace_id_prefill = defaultdict(lambda: None)
        self.trace_inputs_prefill = defaultdict(lambda: None)
        self.trace_output_prefill = defaultdict(lambda: None)
        self.trace_ids_decode = defaultdict(lambda: None)  # {device_sampling_bool: {device_id: trace_id}}
        self.trace_inputs_decode = defaultdict(lambda: None)
        self.trace_output_decode = defaultdict(lambda: None)

        # Adaptive L1 KV cache deferred allocation flags.
        # l1_kv_needs_alloc is True when any model has an l1_kv_window_size > 0
        # OR has use_adaptive_l1_kv_cache=True (N-tier adaptive path).
        # Allocation is triggered once after the decode compile step.
        self.l1_kv_needs_alloc = any(
            getattr(m_args, "l1_kv_window_size", 0) > 0 or getattr(m_args, "use_adaptive_l1_kv_cache", False)
            for m_args in self.model_args
        )
        self.l1_kv_safety_margin = getattr(model_args[0], "l1_kv_safety_margin", 64 * 1024)
        self.l1_kv_min_viable_tokens = getattr(model_args[0], "l1_kv_min_viable_tokens", 64)
        # Used by the no-trace path to detect when the compile iteration has already run.
        self._decode_compile_done = False
        # True after the warmup step (step 2) has run for the improved live-scan path.
        # Not needed for the JSON path (allocation happens immediately after compile).
        self._l1_kv_warmup_done = False

    def _capture_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        kv_cache=None,
        model_id=-1,
    ):
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, page_table=page_table)
        # These matrices will actually be pointing to the whole cos_matrix and sin_matrix that was allocated on device in the RotarySetup class
        tt_rot_mats_prefill_global = host_inputs[1]
        tt_rot_mats_prefill_local = host_inputs[2]
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        logger.info("Done Compiling Model")

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        logger.info("Done Capturing Prefill Trace")
        return trace_id, tt_out_trace, *device_inputs

    def _easy_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        user_id=0,
        last_token_idx=None,
        kv_cache=None,
        model_id=-1,
        prefill_seq_len=None,
        **kwargs,
    ):
        # We are not appending host/device here because we never do device sampling in prefill with TTT
        trace_key = f"{prefill_seq_len}_{model_id}"
        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                prefill_ids,
                page_table=page_table,
                kv_cache=kv_cache,
                model_id=model_id,
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace

        tt_out_trace = self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            prefill_ids,
            page_table=page_table,
            model_id=model_id,
        )

        return tt_out_trace

    def _prefill_forward_trace(
        self,
        trace_id,
        device_inputs,
        tt_out_trace,
        prefill_ids,
        user_id=0,
        page_table=None,
        model_id=-1,
    ):
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, page_table=page_table)
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        device_inputs = copy_host_to_device(
            host_inputs, device_tensors=device_inputs, mesh_device=self.model_args[model_id].mesh_device
        )

        ttnn.execute_trace(self.model_args[model_id].mesh_device, trace_id, cq_id=0, blocking=False)

        return tt_out_trace

    # Note: This function is called by vLLM
    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        **kwargs,
    ):
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        else:
            # Only paged attention is supported for prefill
            enable_trace = False

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size

        # Each model expected to run the same model, safe to use 1st vocab size
        output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        out_list = []
        for idx, user_id in enumerate(empty_slots):
            model_id = user_id // max_batch_size_per_model
            group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
            seq_len = int(prompt_lens[idx])
            last_token_idx = seq_len - 1
            prefill_seq_len = get_padded_prefill_len(seq_len)
            local_kwargs = kwargs.copy()  # Avoid modifying original kwargs

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            # Extracting data for the current user
            # If page_table is not provided, we keep track of the relative/model user_id through group_user_id
            prefill_ids = torch.cat(
                [tokens[idx : idx + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )

            enable_trace_current_prompt = enable_trace and self.model_args[model_id].can_enable_trace(prefill_seq_len)

            logger.info(
                f"Prefill seq len: {prefill_seq_len}, max_prefill_chunk_size: {self.model_args[0].max_prefill_chunk_size}, trace: {enable_trace_current_prompt}"
            )

            page_table_user = (
                self._get_prefill_user_page_table(
                    page_table[idx : idx + 1],
                    kv_cache[model_id],
                    seq_len,
                    trace_enabled=enable_trace_current_prompt,
                    prefill_seq_len=prefill_seq_len,
                )
                if page_table is not None
                else None
            )
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

            # Check if 'pixel_values' exists and index it safely
            if local_kwargs.get("pixel_values", None) is not None:
                local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                if "image_grid_thw" in local_kwargs:
                    local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]

            if enable_trace_current_prompt:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    prefill_seq_len=prefill_seq_len,
                    **local_kwargs,
                )
            else:
                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    **local_kwargs,
                )
            if enable_trace_current_prompt:
                # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
                # We need to do this here, because we can't do this part in forward() if we have trace enabled
                # The reason we can't do it in trace is because we can't pass the correct get_last_token to trace
                logits = self.model[model_id].process_logits_after_prefill_trace(logits, last_token_idx)

            # if data parallel is greater than 1, we need to add logits to out_list and do the processing after all the prefill are done
            # otherwise, we can process the logits after prefill immediately
            if self.data_parallel > 1:
                out_list.append(logits)
            else:
                output_logits[idx] = self.model[model_id].process_output_prefill(
                    logits, last_token_idx=(last_token_idx % 32)
                )
                del logits

        # Process the logits after all the prefill are done in data parallel mode
        if self.data_parallel > 1:
            for idx, out in enumerate(out_list):
                seq_len = int(prompt_lens[idx])
                last_token_idx = seq_len - 1
                user_id = empty_slots[idx]
                model_id = user_id // max_batch_size_per_model

                # Since we give unpadded_seq_len, only the tile containing the last token is returned
                output_logits[idx] = self.model[model_id].process_output_prefill(
                    out, last_token_idx=(last_token_idx % 32)
                )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits

    def prefill_forward_single_user_text(
        self, tokens, page_table, user_id, last_token_idx, kv_cache=None, model_id=-1, **kwargs
    ):
        seq_len = tokens.shape[-1]
        use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
        if use_chunked_prefill:
            """
            Chunked prefill requires paged attention. There are some strange constraints which we must meet:
             - page_table, which is used in SDPA, must match batch size of inputs, which is 1. This is because SDPA
             checks that page table batch dim matches input batch dim. Therefore we must slice the page table for the current user.
             - page_table must also have enough entries in each chunk, so it will be padded with zeros if necessary.
             - chunked_page_table is the slice of the page table for the current chunk. This is used by paged_fill_cache
             to keep it otherwise unaware that it is operating on a chunk.
             - due to the above point, we must always set user_id to 0 for chunked prefill.
            """
            assert page_table is not None, "page_table must be provided for chunked prefill"
            assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
            assert (
                last_token_idx is not None and last_token_idx < seq_len
            ), "last_token_idx must be provided and less than seq_len"
            chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args[model_id].max_prefill_chunk_size)
            block_size = get_block_size(kv_cache)
            last_token_idx_in_chunk = last_token_idx % chunk_size
            # Calculate which chunk contains the last_token_idx
            last_chunk_start = (last_token_idx // chunk_size) * chunk_size
            page_table_user = page_table[user_id : user_id + 1, :]
            # Pad page table to match number of blocks in seq_len
            num_padding_blocks = num_blocks_in_seq(seq_len, block_size) - page_table_user.shape[1]
            page_table_user_padded = torch.cat(
                [page_table_user, torch.zeros(1, num_padding_blocks, dtype=torch.int32)], dim=-1
            )
            CHUNK_USER_ID = 0

            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = chunk_start + chunk_size
                assert (
                    chunk_end <= seq_len
                ), f"Chunk end should be less than seq_len, got chunk_end={chunk_end} and seq_len={seq_len}"
                chunk_tokens = tokens[:, chunk_start:chunk_end]
                chunk_page_table = page_table_user[:, chunk_start // block_size : chunk_end // block_size]

                (
                    chunk_prefill_input,
                    chunk_rot_mats_global_prefill,
                    chunk_rot_mats_local_prefill,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                    **kwargs,
                )
                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=chunk_rot_mats_global_prefill,
                    rot_mats_local=chunk_rot_mats_local_prefill,
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    valid_seq_len=chunk_tokens.shape[-1],
                    get_last_token=(last_token_idx_in_chunk // 32) * 32,
                    kv_cache=kv_cache,
                    **kwargs,
                )

                if chunk_start == last_chunk_start:
                    return tt_logits
                else:
                    del tt_logits
        else:
            (
                prefill_input,
                rot_mats_global_prefill,
                rot_mats_local_prefill,
                page_table_tt,
                _,
            ) = self.model[model_id].prepare_inputs_prefill(
                tokens,
                page_table=page_table,
                **kwargs,
            )

            tt_logits = self.model[model_id].ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global_prefill,
                rot_mats_local=rot_mats_local_prefill,
                user_id=user_id,
                page_table=page_table_tt,
                valid_seq_len=last_token_idx + 1,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )
            return tt_logits

    # Note: This function is called by vLLM
    def decode_forward_text(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params: SamplingParams = None,  # Should be None if not greedy decoding / sampling on device.
    ):
        sampling_on_device = sampling_params is not None

        B = tokens.shape[0]
        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None

        if sampling_on_device:
            if not isinstance(sampling_params.temperature, List):
                sampling_params_list = [sampling_params] * self.data_parallel
            else:
                temperature_chunks = split_list(sampling_params.temperature, self.data_parallel)
                top_k_chunks = split_list(sampling_params.top_k, self.data_parallel)
                top_p_chunks = split_list(sampling_params.top_p, self.data_parallel)

                # Create new SamplingParams objects for each chunk
                sampling_params_list = []
                for i in range(self.data_parallel):
                    new_params = SamplingParams(
                        temperature=temperature_chunks[i], top_k=top_k_chunks[i], top_p=top_p_chunks[i]
                    )
                    sampling_params_list.append(new_params)

            for i in range(self.data_parallel):
                formatted_params = format_sampling_params(
                    sampling_params_list[i], 32
                )  # Sampling needs params padded to 32 regardless of batch_size
                self.model[i].tt_sampling.reset_params(
                    k=formatted_params.top_k,
                    p=formatted_params.top_p,
                    temp=formatted_params.temperature,
                )
        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "sampling_on_device": sampling_on_device,
        }
        if enable_trace:
            tt_decode_output = self._decode_forward_trace_text(**decode_kwargs)
        else:
            tt_decode_output = self._decode_forward_no_trace_text(**decode_kwargs)

        if read_from_device:
            to_host = self.read_decode_output(tt_decode_output)
            return self.process_decode_output_host(to_host, is_tokens=(sampling_params is not None))

        return tt_decode_output

    def _decode_forward_no_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """
        # No-trace path: L1 KV allocation dispatch.
        #
        # Both JSON and live-scan paths follow the SAME sequence:
        #   Step 1 (compile decode): run normally. This raises the L1 bottom-up
        #           watermark past every program's static CB region end.
        #           We mark _decode_compile_done after it returns.
        #   Step 2 (allocate+decode): first call _post_compile_allocate_l1_kv()
        #           (reads from JSON or live scan), THEN run the normal decode body.
        #
        # Why must allocation happen AFTER step 1?
        #   The HEIGHT_SHARDED KV tensor is placed by the L1 bottom-up allocator at
        #   the first free address. Before step 1 runs, that address may be inside a
        #   static CB region of a later program (e.g. embedding at 0..110976). Step 1
        #   runs ALL programs, which registers their CB regions and raises the watermark
        #   above max(cb_region_end) across all programs. Only then is it safe to place
        #   the KV buffer, as the allocator will land it above all CB regions.
        #
        # For the live-scan path, step 2 runs the inlined decode body first to ensure
        # output tensors are alive during the headroom query (more accurate estimate).
        _has_json = bool(getattr(self.model_args[0], "l1_kv_headroom_json", None))
        if self.l1_kv_needs_alloc and self._decode_compile_done:
            if _has_json:
                # JSON path: allocate now (bottom-up watermark is already raised
                # by the compile step). Read headroom from the offline JSON.
                self._post_compile_allocate_l1_kv()
            elif self._l1_kv_warmup_done:
                # Live-scan path: warmup step already ran and scan+alloc done.
                # This branch is a safety net; normally won't be reached.
                self._post_compile_allocate_l1_kv()
            else:
                # Live-scan path, warmup step: run the full decode body FIRST,
                # then scan headroom while output tensors are still alive.
                self._l1_kv_warmup_done = True
                self._decode_compile_done = True  # prevent re-triggering on next call
                tt_logits_warmup = []
                tt_tokens_w = []
                tt_current_pos_w = []
                tt_rot_mat_idxs_w = []
                tt_page_table_w = []
                tt_l1_update_pos_w = []
                tt_l1_write_enabled_w = []
                for i in range(self.data_parallel):
                    user_page_table_w = page_table[i] if page_table is not None else None
                    model_i_w = self.model[i]
                    (
                        tt_tokens_wi,
                        tt_current_pos_wi,
                        tt_rot_mat_idxs_wi,
                        tt_page_table_wi,
                        tt_l1_update_pos_wi,
                        tt_l1_write_enabled_wi,
                    ) = model_i_w.prepare_inputs_decode(tokens[i], current_pos[i], user_page_table_w)
                    tt_tokens_w.append(tt_tokens_wi)
                    tt_current_pos_w.append(tt_current_pos_wi)
                    tt_rot_mat_idxs_w.append(tt_rot_mat_idxs_wi)
                    tt_page_table_w.append(tt_page_table_wi)
                    tt_l1_update_pos_w.append(tt_l1_update_pos_wi)
                    tt_l1_write_enabled_w.append(tt_l1_write_enabled_wi)
                for i in range(self.data_parallel):
                    user_kv_cache_w = kv_cache[i] if kv_cache is not None else None
                    tt_logits_wi = self.model[i].ttnn_decode_forward(
                        tt_tokens_w[i],
                        tt_current_pos_w[i],
                        rot_mat_idxs=tt_rot_mat_idxs_w[i],
                        page_table=tt_page_table_w[i],
                        l1_update_pos=tt_l1_update_pos_w[i],
                        l1_write_enabled=tt_l1_write_enabled_w[i],
                        kv_cache=user_kv_cache_w,
                        sampling_on_device=sampling_on_device,
                    )
                    tt_logits_warmup.append(tt_logits_wi)
                # Output tensors still alive here — scan captures transient top-down buffers.
                self._post_compile_allocate_l1_kv()
                return tt_logits_warmup
        self._decode_compile_done = True

        # T3 / T4 diagnostic checkpoints. Fire only on the first step after
        # _post_compile_allocate_l1_kv stashed a T2 snapshot, then clear so we
        # don't re-log every token. Diff against T2 shows what step 2's pre-decode
        # ops (prepare_inputs_decode, then ttnn_decode_forward internals) allocate.
        _t3_t4_active = getattr(self, "_t2_headroom_map", None) is not None
        if _t3_t4_active:
            mesh_dev = self.model_args[0].mesh_device
            logger.info("[L1 KV checkpoint] === T3: start of step-2 decode body, pre prepare_inputs_decode ===")
            t3_map = mesh_dev.get_l1_headroom_per_core()
            self._log_post_alloc_diff(self._t2_headroom_map, t3_map, tag="T2→T3")

        tt_logits = []

        tt_tokens = []
        tt_current_pos = []
        tt_rot_mat_idxs = []
        tt_page_table = []
        tt_l1_update_pos = []
        tt_l1_write_enabled = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            model_i = self.model[i]
            (
                tt_tokens_i,
                tt_current_pos_i,
                tt_rot_mat_idxs_i,
                tt_page_table_i,
                tt_l1_update_pos_i,
                tt_l1_write_enabled_i,
            ) = model_i.prepare_inputs_decode(tokens[i], current_pos[i], user_page_table)
            tt_tokens.append(tt_tokens_i)
            tt_current_pos.append(tt_current_pos_i)
            tt_rot_mat_idxs.append(tt_rot_mat_idxs_i)
            tt_page_table.append(tt_page_table_i)
            tt_l1_update_pos.append(tt_l1_update_pos_i)
            tt_l1_write_enabled.append(tt_l1_write_enabled_i)

        if _t3_t4_active:
            logger.info("[L1 KV checkpoint] === T4: after prepare_inputs_decode, pre ttnn_decode_forward ===")
            t4_map = self.model_args[0].mesh_device.get_l1_headroom_per_core()
            self._log_post_alloc_diff(self._t2_headroom_map, t4_map, tag="T2→T4")
            self._t2_headroom_map = None  # consume — diagnostics fire once

        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            tt_logits_i = self.model[i].ttnn_decode_forward(
                tt_tokens[i],
                tt_current_pos[i],
                rot_mat_idxs=tt_rot_mat_idxs[i],
                page_table=tt_page_table[i],
                l1_update_pos=tt_l1_update_pos[i],
                l1_write_enabled=tt_l1_write_enabled[i],
                kv_cache=user_kv_cache,
                sampling_on_device=sampling_on_device,
            )
            tt_logits.append(tt_logits_i)

        return tt_logits

    def _post_compile_allocate_l1_kv(self):
        """
        Allocate adaptive L1 KV cache tiers after CB addresses are frozen.

        Headroom source (in priority order):
          1. JSON path  -- ``model_args.l1_kv_headroom_json`` is set:
                          load ``gap_bytes_free_headroom`` from an offline profiling
                          JSON produced by TT_METAL_LOG_L1_CB_MAP.  This is the most
                          accurate source: it captures both CB and mid-step transient
                          top-down allocations.
          2. Live-scan  -- called AFTER a warmup decode step so that output tensors are
                          still alive, capturing more of the transient top-down region
                          than a between-step scan would.
        """
        if not self.l1_kv_needs_alloc:
            return

        mesh_dev = self.model_args[0].mesh_device
        logger.info("[L1 KV checkpoint] === T1: before any KV tier allocation ===")
        live_headroom_map = mesh_dev.get_l1_headroom_per_core()  # always probe, used for diff

        json_path = getattr(self.model_args[0], "l1_kv_headroom_json", None)
        if json_path:
            headroom_map = self._load_headroom_json(json_path)
            logger.info(f"[L1 KV] Using offline headroom JSON: {json_path} ({len(headroom_map)} cores)")
            self._log_headroom_diff(headroom_map, live_headroom_map)
        else:
            headroom_map = live_headroom_map
            logger.info("[L1 KV] Using live headroom scan (improved: mid-step measurement)")

        # DEBUG: print full headroom map sorted by core coordinate
        sorted_headroom = sorted(headroom_map.items(), key=lambda kv: (kv[0][1], kv[0][0]))
        headroom_lines = ", ".join(f"({x},{y}):{b}" for (x, y), b in sorted_headroom)
        logger.info(
            f"[L1 KV] Headroom acquired for {len(headroom_map)} cores. "
            f"Safety margin: {self.l1_kv_safety_margin // 1024} KiB. "
            f"min_viable_tokens: {self.l1_kv_min_viable_tokens}. "
            f"Allocating adaptive L1 KV cache..."
        )
        logger.info(f"[L1 KV] Full headroom map (core: bytes): {headroom_lines}")
        self._log_padding_overhead()

        for model_i in self.model:
            for layer in model_i.layers:
                if hasattr(layer, "attention"):
                    layer.attention.allocate_l1_kv_cache(
                        headroom_map,
                        safety_margin_bytes=self.l1_kv_safety_margin,
                        min_viable_tokens=self.l1_kv_min_viable_tokens,
                    )

        self.l1_kv_needs_alloc = False
        logger.info("[L1 KV] Adaptive L1 KV cache allocation complete.")

        # T2 checkpoint: capture per-bank top_down/cb_end immediately after KV alloc.
        # Diff against T1 tells us exactly which banks the KV tiers landed on and how
        # much top-down space each consumed.
        logger.info("[L1 KV checkpoint] === T2: immediately after all KV tier allocations ===")
        post_alloc_map = mesh_dev.get_l1_headroom_per_core()
        self._log_post_alloc_diff(live_headroom_map, post_alloc_map, tag="T1→T2")
        self._t2_headroom_map = post_alloc_map  # stash for T3 comparison

    @staticmethod
    def _log_post_alloc_diff(t1: dict, t2: dict, tag: str = "Δ") -> None:
        """Headroom change `tag` (e.g. 'T1→T2'). Negative delta = headroom shrank
        (= top-down stack grew). Positive delta on an off-tier core is unexpected."""
        common = sorted(set(t1.keys()) & set(t2.keys()))
        deltas = [(c, t2[c] - t1[c]) for c in common]
        deltas_sorted = sorted(deltas, key=lambda kv: kv[1])
        most_neg = deltas_sorted[:16]
        unchanged = [c for c, d in deltas if d == 0]
        logger.info(
            f"[{tag}] {len(deltas)} cores; {len(unchanged)} unchanged; "
            f"min={deltas_sorted[0][1]:+d} B, max={deltas_sorted[-1][1]:+d} B"
        )
        logger.info(
            f"[{tag}] 16 most-negative (= biggest top-down growth): "
            + ", ".join(f"({x},{y}):{d:+d}" for (x, y), d in most_neg)
        )

    @staticmethod
    def _load_headroom_json(path: str) -> dict:
        """
        Load gap_bytes_free_headroom per core from an offline profiling JSON.

        Expected JSON format (produced by TT_METAL_LOG_L1_CB_MAP profiling tool)::

            {
              "(x,y)": {
                "gap_bytes_free_headroom": 343424,
                ...
              },
              ...
            }

        Returns
        -------
        dict[(int, int), int]
            Mapping from ``(x, y)`` core coordinate to available headroom bytes.
        """
        import json
        import re

        with open(path) as f:
            raw = json.load(f)

        headroom_map = {}
        for key, val in raw.items():
            m = re.match(r"\((\d+),(\d+)\)", key)
            if m:
                headroom_map[(int(m.group(1)), int(m.group(2)))] = int(val["gap_bytes_free_headroom"])

        if not headroom_map:
            raise ValueError(f"[L1 KV] No valid core entries found in headroom JSON: {path}")
        return headroom_map

    @staticmethod
    def _log_headroom_diff(json_map: dict, live_map: dict) -> None:
        """
        Compare offline JSON headroom against a live scan taken at allocation time.

        delta = live - json
          delta < 0 → live reports LESS free space than the JSON promised (cores
                      where some persistent buffer exists now that wasn't there
                      at JSON-capture time, or where the current code path keeps
                      a different intermediate alive). These are the cores most
                      likely to OOM when we try to consume the JSON budget.
          delta > 0 → live reports MORE free space (transient peak from JSON
                      profile no longer in flight; expected and harmless).
        """
        common = sorted(set(json_map.keys()) & set(live_map.keys()))
        if not common:
            logger.warning("[L1 KV diff] No overlapping cores between JSON and live maps")
            return

        deltas = [(core, live_map[core] - json_map[core]) for core in common]
        deltas_sorted = sorted(deltas, key=lambda kv: kv[1])  # ascending: worst (most negative) first

        n_neg = sum(1 for _, d in deltas if d < 0)
        n_zero = sum(1 for _, d in deltas if d == 0)
        n_pos = sum(1 for _, d in deltas if d > 0)
        min_d = deltas_sorted[0][1]
        max_d = deltas_sorted[-1][1]
        avg_d = sum(d for _, d in deltas) // len(deltas)
        json_only = set(json_map.keys()) - set(live_map.keys())
        live_only = set(live_map.keys()) - set(json_map.keys())

        logger.info(
            f"[L1 KV diff] live - json: {n_neg} cores LESS, {n_zero} equal, {n_pos} MORE "
            f"(min={min_d:+d} B, max={max_d:+d} B, avg={avg_d:+d} B); "
            f"json_only={len(json_only)}, live_only={len(live_only)}"
        )
        worst = deltas_sorted[:16]
        worst_str = ", ".join(f"({x},{y}):{d:+d}" for (x, y), d in worst)
        logger.info(f"[L1 KV diff] 16 most-negative deltas (live<json): {worst_str}")
        best = deltas_sorted[-8:]
        best_str = ", ".join(f"({x},{y}):{d:+d}" for (x, y), d in best)
        logger.info(f"[L1 KV diff] 8 most-positive deltas (live>json): {best_str}")
        if json_only:
            logger.info(f"[L1 KV diff] cores present only in JSON: {sorted(json_only)}")
        if live_only:
            logger.info(f"[L1 KV diff] cores present only in live scan: {sorted(live_only)}")

    def _log_padding_overhead(self) -> None:
        """
        Log raw vs padded per-tile cost for the KV dtype so we can quantify how
        much the ``_build_adaptive_l1_memcfg_tiers`` budget formula under-counts.

        Llama 3.1 8B uses bfloat8_b for KV, which stores 1 mantissa byte per
        element plus a shared-exponent byte per 16 elements → 1088 B per 32×32
        tile (instead of the naive 1024 B). The formula in attention.py uses
        elem_bytes=1 with no per-tile exponent term, so it under-counts by
        1088 / 1024 ≈ 6.25%. The OOM messages in the failing run confirm this:
            Tier 1 (1 tile-row × 4 tile-wide):  raw 4096   actual 4352   (+6.25%)
            Tier 2 (2 tile-rows × 4 tile-wide): raw 8192   actual 8704   (+6.25%)
            Tier 3 (4 tile-rows × 4 tile-wide): raw 16384  actual 17408  (+6.25%)
        """
        per_tile_raw_b16 = 32 * 32 * 2
        per_tile_padded_bfp8 = 1088  # 1024 mantissa + 64 exponent bytes
        per_tile_raw_bfp8 = 1024
        overhead_pct_bfp8 = 100 * (per_tile_padded_bfp8 - per_tile_raw_bfp8) / per_tile_raw_bfp8
        logger.info(
            f"[L1 KV padding] per-tile sizes: bfloat16={per_tile_raw_b16} B (no exponent overhead); "
            f"bfloat8_b={per_tile_padded_bfp8} B actual vs {per_tile_raw_bfp8} B in budget formula "
            f"(+{overhead_pct_bfp8:.2f}% under-counted)."
        )
        logger.info(
            f"[L1 KV padding] attention.py:_build_adaptive_l1_memcfg_tiers uses "
            f"`tile_size * head_dim * elem_bytes` which equals raw tile bytes for bfloat16 "
            f"but ignores the exponent bytes for bfloat8_b. If KV dtype is bfloat8_b, the "
            f"per-tile-row budget is under-counted by ~6.25% per layer, accumulating across "
            f"all layers (e.g. 32 layers × ~6% ≈ 16 KiB unaccounted on the highest tier)."
        )

    def _capture_decode_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
    ):
        """
        Captures a trace for the decode_forward method.
        """

        # Compile run
        self._decode_forward_no_trace_text(
            tokens,
            current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_on_device=sampling_on_device,
        )
        logger.info("Done Compiling Model")

        # Post-compile: allocate adaptive L1 KV cache now that all CB addresses are frozen.
        if self.l1_kv_needs_alloc:
            self._post_compile_allocate_l1_kv()

        # Get inputs ready for trace run
        device_inputs = []
        tt_out_trace = []
        trace_ids = {}
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None

            with l1_kv_perf.timed("decode.prepare_inputs_host"):
                host_inputs = self.model[i].prepare_decode_inputs_host(
                    tokens[i], current_pos[i], page_table=user_page_table
                )

            with l1_kv_perf.timed("decode.host_to_device"):
                device_inputs_i = copy_host_to_device(host_inputs[:-1], mesh_device=self.model_args[i].mesh_device)
            device_inputs.append(device_inputs_i)

        for i in range(self.data_parallel):
            trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
            trace_ids[i] = trace_id
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            tt_out_trace.append(
                self.model[i].ttnn_decode_forward(
                    *device_inputs[i],
                    kv_cache=user_kv_cache,
                    sampling_on_device=sampling_on_device,
                )
            )
            ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")
        return trace_ids, tt_out_trace, *device_inputs

    def _decode_forward_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
    ):
        """
        Run decode forward text with tracing
        """
        # The trace is different depending on whether we are doing device sampling or not
        if not self.trace_ids_decode[sampling_on_device]:
            trace_ids, tt_out_trace, *device_inputs = self._capture_decode_trace_text(
                tokens, current_pos, page_table=page_table, kv_cache=kv_cache, sampling_on_device=sampling_on_device
            )
            self.trace_ids_decode[sampling_on_device] = trace_ids
            self.trace_inputs_decode[sampling_on_device] = device_inputs
            self.trace_output_decode[sampling_on_device] = tt_out_trace

        reset_inputs = not sampling_on_device
        if self.prev_page_table is None or any(
            not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table)
        ):
            reset_inputs = True
            self.prev_page_table = page_table

        if reset_inputs:
            for i in range(self.data_parallel):
                user_page_table = page_table[i] if page_table is not None else None
                with l1_kv_perf.timed("decode.prepare_inputs_host"):
                    host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)

                with l1_kv_perf.timed("decode.host_to_device"):
                    copy_host_to_device(
                        host_tensors=host_inputs_i[:-1],
                        device_tensors=self.trace_inputs_decode[sampling_on_device][i],
                    )

        for i, trace_id in self.trace_ids_decode[sampling_on_device].items():
            ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)

        return self.trace_output_decode[sampling_on_device]

    def _prefill_forward_single_user(
        self,
        vision_images,
        vision_mask,
        tokens,
        xattn_caches,
        user_id,
        total_len,
        prefill_len,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        model_id=-1,
    ):
        """
        Performs vision encode step then text prefill.
        Returns (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits)
        """
        B = tokens.shape[0]
        last_token_idx = prefill_len - 1

        text_only_inference = vision_images is None
        if not text_only_inference:
            (
                vision_tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
            ) = self.model[model_id].compute_vision_tokens_masks(
                batch_images=[vision_images],
                batch_masks=[vision_mask],
                total_len=total_len,
                prefill_len=prefill_len,
            )

            if cross_page_table is not None:
                num_vision_tokens = vision_tokens.shape[2]
                cross_page_table = self._get_prefill_user_page_table(cross_page_table, kv_cache, num_vision_tokens)
        else:
            (
                vision_tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
            ) = (None, None, None, None, None)

        if page_table is not None:
            page_table = self._get_prefill_user_page_table(page_table, kv_cache, prefill_len)

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            rot_mats,
            tt_page_table,
            tt_cross_page_table,
        ) = self.model[model_id].prepare_inputs_prefill(
            tokens,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            prefill_len=prefill_len,
            page_table=page_table,
            cross_page_table=cross_page_table,
            text_only_inference=text_only_inference,
        )

        tt_logits = self.model[model_id].ttnn_prefill_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            xattn_caches,
            rot_mats,
            user_id,
            vision_tokens,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            get_last_token=(last_token_idx // 32) * 32,
            cross_page_table=tt_cross_page_table,
            text_only_inference=text_only_inference,
        )

        del tt_page_table
        del tt_cross_page_table

        return (
            xattn_caches,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            tt_logits,
        )

    # Note: This function is called by vLLM
    def prefill_forward(
        self,
        vision_images,
        vision_masks,
        tokens,
        xattn_caches,
        total_lens,
        prompt_lens,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        empty_slots=None,
        **kwargs,
    ):
        if (self.model_args[0].checkpoint_type == CheckpointType.HuggingFace) and (
            not self.model_args[0].is_llama_vision()
        ):
            logits = self.prefill_forward_text(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                pixel_values=vision_images,
                **kwargs,
            )

            return logits, None, None, None, None

        else:
            (
                output_logits,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
            ) = self.prefill_forward_llama_vision(
                vision_images,
                vision_masks,
                tokens,
                xattn_caches,
                total_lens,
                prompt_lens,
                page_table=page_table,
                kv_cache=kv_cache,
                cross_page_table=cross_page_table,
                empty_slots=empty_slots,
            )

            return (
                output_logits,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
            )

    # Note: This function is called by vLLM
    def prefill_forward_llama_vision(
        self,
        vision_images,
        vision_masks,
        tokens: torch.Tensor,
        xattn_caches,
        total_lens,
        prompt_lens,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        empty_slots=None,
    ):
        """
        Batched version of _prefill_forward_single_user for vision model.
        """
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        if cross_page_table is not None:
            assert isinstance(cross_page_table, torch.Tensor), "cross_page_table mush be torch.Tensor"

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size

        output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)

        out_list = []
        prefill_output_xattn_masks = []
        prefill_output_full_text_row_masked_out_masks = []
        decode_output_xattn_masks = []
        decode_output_full_text_row_masked_out_masks = []

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        for idx, user_id in enumerate(empty_slots):
            model_id = user_id // max_batch_size_per_model
            group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
            seq_len = int(prompt_lens[idx])

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            user_page_table = page_table[idx : idx + 1] if page_table is not None else None
            user_cross_page_table = cross_page_table[idx : idx + 1] if kv_cache is not None else None
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None
            model_xattn_cache = xattn_caches[model_id] if xattn_caches is not None else None

            (
                model_xattn_cache,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
                logits,
            ) = self._prefill_forward_single_user(
                vision_images=vision_images[idx],
                vision_mask=vision_masks[idx],
                tokens=tokens[idx : idx + 1, :seq_len],  # Keep batch dimension
                xattn_caches=model_xattn_cache,
                user_id=group_user_id,
                total_len=total_lens[idx],
                prefill_len=seq_len,
                page_table=user_page_table,
                kv_cache=model_kv_cache,
                cross_page_table=user_cross_page_table,
                model_id=model_id,
            )

            if xattn_caches is not None:
                xattn_caches[model_id] = model_xattn_cache

            out_list.append(logits)
            prefill_output_xattn_masks.append(prefill_cross_attention_masks)
            prefill_output_full_text_row_masked_out_masks.append(prefill_full_text_row_masked_out_mask)
            decode_output_xattn_masks.append(decode_cross_attention_masks)
            decode_output_full_text_row_masked_out_masks.append(decode_full_text_row_masked_out_mask)

        # We gather prefill output at the end of prefill to reduce unnecessary device sync
        for idx, user_id in enumerate(empty_slots):
            model_id = user_id // max_batch_size_per_model

            last_token_idx = prompt_lens[idx] - 1
            output_logits[idx] = self.model[model_id].process_output_prefill(
                out_list[idx], 1, last_token_idx=(last_token_idx % 32)
            )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return (
            output_logits,
            prefill_output_xattn_masks,
            prefill_output_full_text_row_masked_out_masks,
            decode_output_xattn_masks,
            decode_output_full_text_row_masked_out_masks,
        )

    # Note: This function is called by vLLM
    def decode_forward_llama_vision(
        self,
        start_pos,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        enable_trace=True,
        read_from_device=True,
    ):
        B = tokens.shape[0]
        data_parallel = min(B, self.data_parallel)
        batch_per_device = B // data_parallel
        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        prefill_cross_attention_masks = [
            prefill_cross_attention_masks[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        prefill_full_text_row_masked_out_mask = [
            prefill_full_text_row_masked_out_mask[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        decode_cross_attention_masks = [
            decode_cross_attention_masks[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        decode_full_text_row_masked_out_mask = [
            decode_full_text_row_masked_out_mask[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None
        cross_page_table = (
            torch.chunk(cross_page_table, self.data_parallel, 0) if cross_page_table is not None else None
        )

        decode_kwargs = {
            "position_id": start_pos,
            "tokens": tokens,
            "prefill_cross_attention_masks": prefill_cross_attention_masks,
            "prefill_full_text_row_masked_out_mask": prefill_full_text_row_masked_out_mask,
            "decode_cross_attention_masks": decode_cross_attention_masks,
            "decode_full_text_row_masked_out_mask": decode_full_text_row_masked_out_mask,
            "xattn_caches": xattn_caches,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "cross_page_table": cross_page_table,
        }
        if enable_trace:
            tt_logits = self._easy_trace(**decode_kwargs)
        else:
            tt_logits = self._decode_forward_no_trace(**decode_kwargs)

        if read_from_device:
            to_host = self.read_decode_output(tt_logits)
            return self.process_decode_output_host(to_host)
        else:
            return tt_logits

    def decode_forward(
        self,
        start_pos,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        enable_trace=True,
        read_from_device=True,
    ):
        if (self.model_args[0].checkpoint_type == CheckpointType.HuggingFace) and (
            not self.model_args[0].is_llama_vision()
        ):
            return self.decode_forward_text(
                tokens,
                start_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=kv_cache,
            )
        else:
            return self.decode_forward_llama_vision(
                start_pos,
                tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
                xattn_caches,
                page_table,
                kv_cache,
                cross_page_table,
                enable_trace,
                read_from_device,
            )

    # Note: This function is called by vLLM
    def read_decode_output(self, tt_out, async_read=False):
        """
        Input tt_out is a list of ttnn device tensors
        """
        if not async_read:
            with l1_kv_perf.timed("decode.output_readback"):
                return [out.cpu() for out in tt_out]

        host_outputs = []
        read_events = []
        for i in range(self.data_parallel):
            host_outputs.append(tt_out[i].cpu(blocking=False))
            read_events.append(ttnn.record_event(self.model[i].mesh_device, 0))

        return host_outputs, read_events

    # Note: This function is called by vLLM
    def process_decode_output_host(self, tt_out, is_tokens=False):
        """
        Converts the input ttnn host tensors to a torch tensor.
        The input can be logits (if is_tokens=False) or tokens (if is_tokens=True).
        """
        max_batch_size_per_model = self.model_args[0].max_batch_size

        with l1_kv_perf.timed("decode.output_postprocess"):
            logits = []
            for i in range(self.data_parallel):
                logits_i = self.model[i].process_output_decode(
                    tt_out[i], max_batch_size_per_model, S=1, is_tokens=is_tokens
                )
                logits.append(logits_i)

            return torch.cat(logits, 0)

    def _decode_forward_no_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """

        # forward_decode should be traced callable
        # decorator does compilation, capture, execute
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rot_mats = []
        tt_page_table = []
        tt_cross_page_table = []

        for i in range(self.data_parallel):
            B, S = tokens[i].shape
            assert S == 1

            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rot_mats_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_inputs_decode(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rot_mats.append(tt_rot_mats_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        tt_logits = []
        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            tt_logits_i = self.model[i].ttnn_decode_forward(
                tt_h[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
            tt_logits.append(tt_logits_i)

        return tt_logits

    def _capture_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Captures a trace for the decode_forward method.
        """
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rot_mats = []
        tt_page_table = []
        tt_cross_page_table = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rot_mats_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_inputs_decode(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rot_mats.append(tt_rot_mats_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        # Compile run
        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            # tt_logits_rm unused later, no need to make a list
            tt_logits_rm = self.model[i].ttnn_decode_forward(
                tt_h[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
        logger.info("Done Compiling Model")

        # Get inputs ready for trace run
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rope_id = []
        tt_page_table = []
        tt_cross_page_table = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rope_id_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_decode_inputs_host(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rope_id_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = copy_host_to_device(
                (
                    tt_h_i,
                    tt_xattn_mask_i,
                    tt_full_text_mask_expand_1NSH_i,
                    tt_full_text_mask_expand_11SD_i,
                    tt_position_id_i,
                    tt_rope_id_i,
                    tt_page_table_i,
                    tt_cross_page_table_i,
                ),
                mesh_device=self.model_args[i].mesh_device,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rope_id.append(tt_rope_id_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        tt_h_trace_input = tt_h

        tt_logits_rm = []
        trace_ids = {}
        # Do on-device transformations of inputs before forward
        for i in range(self.data_parallel):
            trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
            trace_ids[i] = trace_id
            B = tokens[i].shape[0]
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            (
                tt_h_transform,
                tt_rot_mats,
                tt_xattn_mask_transform,
                tt_full_text_mask_expand_1NSH_transform,
                tt_full_text_mask_expand_11SD_transform,
            ) = self.model[i].transform_decode_inputs_device(
                tt_h[i],
                tt_rope_id[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                B=B,
            )

            tt_logits_rm_i = self.model[i].ttnn_decode_forward(
                tt_h_transform,
                tt_xattn_mask_transform,
                tt_full_text_mask_expand_1NSH_transform,
                tt_full_text_mask_expand_11SD_transform,
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats,
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
            tt_logits_rm.append(tt_logits_rm_i)
            ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return (
            trace_ids,
            tt_logits_rm,
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        )

    def _decode_forward_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        page_table,
        cross_page_table,
        trace_ids,
        trace_logits_rm,
        trace_h,
        trace_xattn_mask,
        trace_full_text_mask_expand_1NSH,
        trace_full_text_mask_expand_11SD,
        trace_position_id,
        trace_rope_id,
        trace_page_table,
        trace_cross_page_table,
    ):
        """
        Executes the trace for the decode_forward method but does not read back outputs.
        """
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_full_text_mask_expand_11SD,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ) = self.model[i].prepare_decode_inputs_host(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            copy_host_to_device(
                host_tensors=(
                    tt_h,
                    tt_xattn_mask,
                    tt_full_text_mask_expand_1NSH,
                    tt_full_text_mask_expand_11SD,
                    tt_position_id,
                    tt_rope_id,
                    tt_page_table,
                    tt_cross_page_table,
                ),
                device_tensors=(
                    trace_h[i],
                    trace_xattn_mask[i],
                    trace_full_text_mask_expand_1NSH[i],
                    trace_full_text_mask_expand_11SD[i],
                    trace_position_id[i],
                    trace_rope_id[i],
                    trace_page_table[i],
                    trace_cross_page_table[i],
                ),
            )
        for i, trace_id in trace_ids.items():
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_logits_rm

    def _easy_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_ids"):
            (
                trace_ids,
                tt_logits_rm,
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_full_text_mask_expand_11SD,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ) = self._capture_trace(
                position_id,
                tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
                xattn_caches,
                page_table=page_table,
                kv_cache=kv_cache,
                cross_page_table=cross_page_table,
            )
            self.trace_ids = trace_ids
            self.trace_inputs = {
                "tt_h": tt_h,
                "tt_xattn_mask": tt_xattn_mask,
                "tt_full_text_mask_expand_1NSH": tt_full_text_mask_expand_1NSH,
                "tt_full_text_mask_expand_11SD": tt_full_text_mask_expand_11SD,
                "tt_position_id": tt_position_id,
                "tt_rope_id": tt_rope_id,
                "tt_page_table": tt_page_table,
                "tt_cross_page_table": tt_cross_page_table,
            }
            self.trace_outputs = {
                "tt_logits_rm": tt_logits_rm,
            }

        trace_logits_rm = self._decode_forward_trace(
            position_id,
            tokens,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            page_table,
            cross_page_table,
            self.trace_ids,
            self.trace_outputs["tt_logits_rm"],
            self.trace_inputs["tt_h"],
            self.trace_inputs["tt_xattn_mask"],
            self.trace_inputs["tt_full_text_mask_expand_1NSH"],
            self.trace_inputs["tt_full_text_mask_expand_11SD"],
            self.trace_inputs["tt_position_id"],
            self.trace_inputs["tt_rope_id"],
            self.trace_inputs["tt_page_table"],
            self.trace_inputs["tt_cross_page_table"],
        )

        return trace_logits_rm

    def generate(
        self,
        vision_images,
        vision_mask,
        prompt_tokens,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        # Do initial prefill
        prefill_len = len(prompt_tokens)
        total_len = prefill_len + max_gen_len  # Prepares mask for full length of output

        prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long).reshape(1, -1)  # B, S
        # Suboptimal to allocate caches every time
        model_id = 0
        xattn_caches = self.model[model_id].setup_cache(self.model_args[model_id].max_batch_size)
        (
            xattn_caches,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            logits,
        ) = self._prefill_forward_single_user(
            vision_images,
            vision_mask,
            prompt_tokens_tensor,
            xattn_caches,
            user_id=0,
            total_len=total_len,
            prefill_len=prefill_len,
            model_id=model_id,
        )

        last_token_idx = prefill_len - 1
        logits = self.model[model_id].process_output_prefill(logits, 1, last_token_idx=(last_token_idx % 32))
        logits = logits.view(1, 1, self.model_args[model_id].vocab_size)

        prefill_output_xattn_masks = [[] for _ in range(self.data_parallel)]
        prefill_output_full_text_row_masked_out_masks = [[] for _ in range(self.data_parallel)]
        decode_output_xattn_masks = [[] for _ in range(self.data_parallel)]
        decode_output_full_text_row_masked_out_masks = [[] for _ in range(self.data_parallel)]

        prefill_output_xattn_masks[model_id].append(prefill_cross_attention_masks)
        prefill_output_full_text_row_masked_out_masks[model_id].append(prefill_full_text_row_masked_out_mask)
        decode_output_xattn_masks[model_id].append(decode_cross_attention_masks)
        decode_output_full_text_row_masked_out_masks[model_id].append(decode_full_text_row_masked_out_mask)

        def sample(logits):
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            decoder = self.tokenizer or self.processor
            return next_token, decoder.decode(next_token.tolist())

        next_token, text = sample(logits)

        yield TokenResult(
            token=next_token[0].item(),
            text=text,
        )

        for gen_idx in range(max_gen_len - 1):
            position_id = torch.tensor([prefill_len + gen_idx])
            next_token_tensor = next_token.reshape(1, 1)  # B, S

            logits = self.decode_forward(
                position_id,
                next_token_tensor,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
                [xattn_caches],
                enable_trace=False,
            )
            next_token, text = sample(logits)
            yield TokenResult(
                token=next_token[0].item(),
                text=text,
            )

    def chat_completion(
        self,
        messages,
        temperature=0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        model_id = 0
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model[model_id].configuration.max_seq_len:
            max_gen_len = self.model[model_id].configuration.max_seq_len - 1

        encoder = self.processor or self.tokenizer
        model_input = encoder.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        vision_images = extract_images_from_messages(messages) or None
        vision_mask = None
        if vision_images is not None:
            vision_mask = create_vision_mask(model_input["input_ids"][0], encoder.image_token_id) or None

        tokens = []

        stop_reason = None
        for result in self.generate(
            vision_images=vision_images,
            vision_mask=vision_mask,
            prompt_tokens=model_input["input_ids"][0],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)
            if result.text == "<|eot_id|>":
                stop_reason = StopReason.end_of_turn
            elif result.text == "<|eom_id|>":
                stop_reason = StopReason.end_of_message

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        decoder = self.tokenizer or self.processor
        message = decoder.decode(tokens, skip_special_tokens=True)

        return CompletionMessage(message)

    def text_completion(
        self,
        content,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        """Supports only vision models at the moment"""
        model_id = 0
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model[model_id].configuration.max_seq_len:
            max_gen_len = self.model[model_id].configuration.max_seq_len - 1

        vision_images = []
        image_token = getattr(self.processor, "image_token", None) or getattr(self.tokenizer, "image_token", None)
        text = encode_content(content, vision_images, image_token)
        vision_images = vision_images or None
        model_input = self.processor(text=text, images=vision_images, add_special_tokens=False)
        vision_mask = None
        if vision_images is not None:
            vision_mask = create_vision_mask(model_input["input_ids"][0], self.processor.image_token_id) or None

        tokens = []

        for result in self.generate(
            vision_images=vision_images,
            vision_mask=vision_mask,
            prompt_tokens=model_input["input_ids"],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)

        decoder = self.tokenizer or self.processor
        generation = decoder.decode(tokens, skip_special_tokens=True)

        return generation

    def _get_prefill_user_page_table(
        self, page_table, kv_cache, prefill_len, trace_enabled=False, prefill_seq_len=None
    ):
        # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly
        block_size = get_block_size(kv_cache)
        num_blocks = 0
        if trace_enabled:
            num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
        else:
            num_blocks = num_blocks_in_seq(prefill_len, block_size)
        if trace_enabled:
            if page_table.shape[1] < num_blocks:
                # If page table is too short, pad it with -1
                padding = torch.ones(1, num_blocks - page_table.shape[1], dtype=torch.int32) * -1
                page_table = torch.cat([page_table, padding], dim=1)
        return page_table[:, :num_blocks]

    ## Destructor

    def __del__(self):
        # Workaround for issue #19052
        if self.data_parallel > 1:
            for m in self.model:
                ttnn.close_mesh_device(m.mesh_device)

        if hasattr(super(Generator, self), "__del__"):
            super().__del__()


def create_submeshes(mesh_device, data_parallel):
    if not isinstance(mesh_device, ttnn.MeshDevice) or data_parallel == 1:
        return [mesh_device]

    num_rows, num_cols = mesh_device.shape
    num_devices = num_rows * num_cols
    assert num_devices % data_parallel == 0, f"Unsupported device split: {num_devices} devices, {data_parallel} groups"

    if num_rows == 8 and num_cols == 4 and num_cols % data_parallel == 0:
        submeshes = mesh_device.create_submeshes(ttnn.MeshShape(num_rows, num_cols // data_parallel))
        for submesh in submeshes:
            submesh.reshape(ttnn.MeshShape(1, num_devices // data_parallel))
        return submeshes

    return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))
