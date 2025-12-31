import ttnn
import torch
import os

device_id = 0
device = ttnn.open_device(device_id=device_id)

try:
    print("Running Matmul with Sharded Output (Retry)...")

    # Reducing N to 512 to try to satisfy the constraints derived from the error
    # Error was: "Shard width 512 must match physical width 1024"
    # This suggests maybe it defaulted to half? Or maybe I should specificy 512.
    M, K, N = 1024, 1024, 512

    a_torch = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    b_torch = torch.randn((1, 1, K, N), dtype=torch.bfloat16)

    a = ttnn.from_torch(a_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Use a grid we know fits
    # 8x8 = 64 cores.
    # M=1024. 1024/64 = 16.
    grid_size = ttnn.CoreCoord(8, 8)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))})

    shard_height = 16
    shard_width = N

    shard_spec = ttnn.ShardSpec(grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    print(f"Requesting Output Config: {mem_config}")

    out = ttnn.matmul(a, b, memory_config=mem_config)

    print("\n--- Result ---")
    print(f"Output Storage: {out.storage_type()}")
    if out.memory_config().is_sharded():
        print(f"Output Shard Spec Grid: {out.memory_config().shard_spec().grid}")
    else:
        print("Output is NOT sharded.")

except Exception as e:
    print(f"Error during execution: {e}")

finally:
    ttnn.close_device(device)
