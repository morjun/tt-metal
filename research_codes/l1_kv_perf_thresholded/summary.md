# L1 KV Decode Breakdown

| Metric | Dual-source avg ms | DRAM-only avg ms | Delta ms |
| --- | ---: | ---: | ---: |
| `decode.prepare_inputs_host` | 0.529 | 0.304 | +0.225 |
| `decode.host_to_device` | 0.065 | 0.055 | +0.009 |
| `decode.transform_inputs_device` | 44.753 | 44.508 | +0.245 |
| `decode.model_forward` | 1027.119 | 1005.427 | +21.691 |
| `decode.dram_kv_write` | 51.574 | 51.533 | +0.041 |
| `decode.sdpa_call` | 119.305 | 119.436 | -0.131 |
| `decode.output_readback` | 17.690 | 17.680 | +0.010 |
| `decode.output_postprocess` | 0.177 | 0.167 | +0.010 |
| `decode.expected_l1_hit_ratio` | 791.667 | - | - |
