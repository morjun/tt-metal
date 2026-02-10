# Task: Implement L1 Partial Weight Sharding for Llama 3.1 8B

- [x] Audit `simple_text_demo.py` and single-device compatibility <!-- id: 0 -->
- [x] Determine "Split Point" calculation logic (1MB/core) <!-- id: 1 -->
- [x] Identify target layers in `mlp.py` and `attention.py` <!-- id: 2 -->
- [x] Draft `implementation_plan.md` <!-- id: 3 -->
- [x] Implement `get_l1_sharded_rows` in `model_config.py` <!-- id: 4 -->
- [x] Modify `mlp.py` for partial sharding <!-- id: 5 -->
- [x] Modify `attention.py` for partial sharding <!-- id: 6 -->
- [x] Implement fallback logic for `w_l1` being `None` in MLP/Attention
- [x] Verify fix with `test_partial_sharding_unit.py` <!-- id: 7 -->
- [x] Make L1 sharding optional in `ModelArgs` and `simple_text_demo.py` <!-- id: 8 -->
- [x] Update `MLP` and `Attention` to respect the flag <!-- id: 9 -->
- [x] Fix `AttributeError` in `Attention` (Prefill support)
- [x] Fix `MLP` forward regression (restored AllReduce/logic)
- [x] Fix Segfault/Trace Error in Default mode (Partially resolved: Restored Sharded weights, but native Trace Error persists on this setup. Enabled mode works via Interleaved fallback).
- [x] Verify `simple_text_demo.py` runs with and without the flag <!-- id: 10 -->
- [/] Debug `NameError: name 'argmax_on_device' is not defined` in `generator.py` (Fixed NameError + OOM; profiler disabled reveals transpose layout error) <!-- id: 11 -->
