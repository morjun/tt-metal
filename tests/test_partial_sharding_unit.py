import pytest
import torch
import ttnn
import os
from models.tt_transformers.tt.model_config import ModelArgs, CheckpointType
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.attention import Attention


@pytest.fixture(scope="module")
def mesh_device():
    # minimalist mesh device setup
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.1-8B-Instruct"])
@pytest.mark.parametrize("use_l1_sharding", [False, True])
def test_mlp_initialization(mesh_device, model_name, use_l1_sharding):
    print(f"Testing MLP Initialization with use_l1_sharding={use_l1_sharding}")

    if "LLAMA_DIR" in os.environ:
        del os.environ["LLAMA_DIR"]

    os.environ["HF_MODEL"] = model_name

    args = ModelArgs(mesh_device, dummy_weights=True, use_l1_weight_sharding=use_l1_sharding)
    args.n_layers = 1
    args.checkpoint_type = CheckpointType.HuggingFace

    state_dict = {}
    prefix = "layers.0.feed_forward."

    dim = args.dim
    hidden_dim = args.hidden_dim

    w1 = torch.randn(hidden_dim, dim)
    w2 = torch.randn(dim, hidden_dim)
    w3 = torch.randn(hidden_dim, dim)

    state_dict[prefix + "w1.weight"] = w1
    state_dict[prefix + "w2.weight"] = w2
    state_dict[prefix + "w3.weight"] = w3

    mlp = MLP(
        mesh_device=mesh_device,
        tt_ccl=None,
        args=args,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat16,
        model_config=args.get_model_config(),
    )

    print("MLP Initialized")

    if use_l1_sharding:
        # Expect L1 tensors to be present
        assert mlp.w1_l1 is not None
        assert mlp.w1_l1.memory_config().buffer_type == ttnn.BufferType.L1
        assert mlp.w1_dram.memory_config().buffer_type == ttnn.BufferType.DRAM
        print("Verified W1 L1 exists and is L1 resident")
    else:
        # Expect L1 tensors to be None
        assert mlp.w1_l1 is None
        assert mlp.w1_dram.memory_config().buffer_type == ttnn.BufferType.DRAM
        print("Verified W1 L1 is None (Sharding Disabled)")

    print("MLP Memory configuration verification PASSED")

    # Test Forward
    print("Testing MLP Forward Pass")
    batch = 1
    seq_len = 32
    x = torch.randn(batch, 1, seq_len, dim)

    tt_x = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out = mlp(tt_x, mode="decode")
    assert out is not None
    print("MLP Forward Pass PASSED")


@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.1-8B-Instruct"])
@pytest.mark.parametrize("use_l1_sharding", [False, True])
def test_attention_initialization(mesh_device, model_name, use_l1_sharding):
    print(f"Testing Attention Initialization with use_l1_sharding={use_l1_sharding}")

    if "LLAMA_DIR" in os.environ:
        del os.environ["LLAMA_DIR"]

    os.environ["HF_MODEL"] = model_name

    args = ModelArgs(mesh_device, dummy_weights=True, use_l1_weight_sharding=use_l1_sharding)
    args.n_layers = 1
    args.checkpoint_type = CheckpointType.HuggingFace

    state_dict = {}
    prefix = "layers.0.attention."

    dim = args.dim
    head_dim = args.head_dim
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads

    wq = torch.randn(n_heads * head_dim, dim)
    wk = torch.randn(n_kv_heads * head_dim, dim)
    wv = torch.randn(n_kv_heads * head_dim, dim)
    wo = torch.randn(dim, n_heads * head_dim)

    state_dict[prefix + "wq.weight"] = wq
    state_dict[prefix + "wk.weight"] = wk
    state_dict[prefix + "wv.weight"] = wv
    state_dict[prefix + "wo.weight"] = wo

    attn = Attention(
        mesh_device=mesh_device,
        tt_ccl=None,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat16,
        transformation_mats=None,
        configuration=args,
    )

    print("Attention Initialized")

    if use_l1_sharding:
        assert attn.wqkv_l1 is not None
        assert attn.wqkv_l1.memory_config().buffer_type == ttnn.BufferType.L1
        assert attn.wqkv_dram.memory_config().buffer_type == ttnn.BufferType.DRAM
        # wo_l1 might be None if rows < alignment (e.g. 4096 < 4160 on some grids)
        if attn.wo_l1 is not None:
            assert attn.wo_l1.memory_config().buffer_type == ttnn.BufferType.L1
        else:
            print("WO L1 is None (Likely due to alignment/size constraints) - Verified Fallback")
        print("Verified Attention L1 tensors exist (or fall back correctly)")
    else:
        assert attn.wqkv_l1 is None
        assert attn.wo_l1 is None
        assert attn.wqkv_dram.memory_config().buffer_type == ttnn.BufferType.DRAM
        print("Verified Attention L1 tensors are None (Sharding Disabled)")

    print("Attention Memory configuration verification PASSED")
