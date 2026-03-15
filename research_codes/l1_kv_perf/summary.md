# L1 KV Decode Breakdown


| Metric                           | Dual-source avg ms | DRAM-only avg ms | Delta ms |
| -------------------------------- | ------------------ | ---------------- | -------- |
| `decode.prepare_inputs_host`     | 0.496              | 0.303            | +0.193   |
| `decode.host_to_device`          | 0.072              | 0.054            | +0.018   |
| `decode.transform_inputs_device` | 44.846             | 44.542           | +0.304   |
| `decode.model_forward`           | 1071.715           | 1006.918         | +64.797  |
| `decode.dram_kv_write`           | 51.716             | 51.654           | +0.061   |
| `decode.l1_clone_path`           | 46.012             | -                | -        |
| `decode.l1_kv_write`             | 50.645             | -                | -        |
| `decode.sdpa_call`               | 119.856            | 119.568          | +0.288   |
| `decode.output_readback`         | 17.686             | 17.712           | -0.026   |
| `decode.output_postprocess`      | 0.170              | 0.166            | +0.004   |
| `decode.expected_l1_hit_ratio`   | 791.667            | -                | -        |
