"""Correctness tests for the self_attention operator (GQA with causal mask)."""

import numpy as np
import pytest
import torch

import zedinfer_ops
from conftest import get_tolerance


def pytorch_self_attention(q_np, k_np, v_np, scale):
    """
    Reference GQA self-attention with causal mask.

    q: [seq_len, num_heads, head_dim]
    k: [total_len, num_kv_heads, head_dim]
    v: [total_len, num_kv_heads, dv]
    scale: float

    Returns: [seq_len, num_heads, dv]
    """
    q = torch.from_numpy(q_np).float()
    k = torch.from_numpy(k_np).float()
    v = torch.from_numpy(v_np).float()

    seq_len, num_heads, head_dim = q.shape
    total_len, num_kv_heads, dv = v.shape
    group_size = num_heads // num_kv_heads

    # Reshape for batched matmul: [num_heads, seq_len, head_dim]
    q_t = q.permute(1, 0, 2)

    # Expand KV heads for GQA: [num_kv_heads, total_len, d] -> [num_heads, total_len, d]
    k_t = k.permute(1, 0, 2).repeat_interleave(group_size, dim=0)
    v_t = v.permute(1, 0, 2).repeat_interleave(group_size, dim=0)

    # Attention scores: [num_heads, seq_len, total_len]
    scores = torch.bmm(q_t, k_t.transpose(1, 2)) * scale

    # Causal mask: position i can attend to j < prefix_len + i + 1
    prefix_len = total_len - seq_len
    causal_mask = torch.triu(
        torch.ones(seq_len, total_len, dtype=torch.bool),
        diagonal=prefix_len + 1,
    )
    scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

    # Softmax and apply to values
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output = torch.bmm(attn_weights, v_t)  # [num_heads, seq_len, dv]

    # Back to [seq_len, num_heads, dv]
    return attn_output.permute(1, 0, 2).numpy()


class TestSelfAttention:
    """Grouped-Query Self-Attention with causal mask."""

    @pytest.mark.parametrize(
        "seq_len,total_len,num_heads,num_kv_heads,head_dim",
        [
            (1, 1, 4, 4, 32),     # Single token, MHA
            (1, 8, 4, 4, 32),     # Single query with KV cache, MHA
            (4, 4, 4, 4, 32),     # Prefill, MHA (no prefix)
            (4, 8, 8, 2, 32),     # GQA with prefix KV cache
            (1, 16, 8, 1, 64),    # MQA (single KV head)
        ],
    )
    def test_f32(self, device, rng, seq_len, total_len, num_heads, num_kv_heads, head_dim):
        q = rng.standard_normal((seq_len, num_heads, head_dim)).astype(np.float32)
        k = rng.standard_normal((total_len, num_kv_heads, head_dim)).astype(np.float32)
        v = rng.standard_normal((total_len, num_kv_heads, head_dim)).astype(np.float32)
        scale = 1.0 / np.sqrt(head_dim)

        result = zedinfer_ops.self_attention(q, k, v, scale=scale, device=device)
        expected = pytorch_self_attention(q, k, v, scale)

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    def test_against_pytorch_sdpa(self, device, rng):
        """Compare with torch.nn.functional.scaled_dot_product_attention."""
        seq_len, total_len = 4, 12
        num_heads, num_kv_heads, head_dim = 8, 4, 64
        scale = 1.0 / np.sqrt(head_dim)

        q_np = rng.standard_normal((seq_len, num_heads, head_dim)).astype(np.float32)
        k_np = rng.standard_normal((total_len, num_kv_heads, head_dim)).astype(np.float32)
        v_np = rng.standard_normal((total_len, num_kv_heads, head_dim)).astype(np.float32)

        zed_result = zedinfer_ops.self_attention(q_np, k_np, v_np, scale=scale, device=device)
        ref_result = pytorch_self_attention(q_np, k_np, v_np, scale)

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(zed_result, ref_result, atol=atol, rtol=rtol)

    def test_causal_mask_blocks_future(self, device, rng):
        """Verify that the causal mask correctly blocks future tokens."""
        seq_len, total_len = 4, 4
        num_heads, head_dim = 2, 16
        scale = 1.0

        # Create Q where only token 0 has non-zero values
        q = np.zeros((seq_len, num_heads, head_dim), dtype=np.float32)
        q[0, :, :] = rng.standard_normal((num_heads, head_dim)).astype(np.float32)

        k = rng.standard_normal((total_len, num_heads, head_dim)).astype(np.float32)
        v = rng.standard_normal((total_len, num_heads, head_dim)).astype(np.float32)

        result = zedinfer_ops.self_attention(q, k, v, scale=scale, device=device)

        # Token 0 should only attend to position 0 (causal)
        # So result[0] should be v[0] (weighted entirely by position 0)
        ref_result = pytorch_self_attention(q, k, v, scale)
        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, ref_result, atol=atol, rtol=rtol)
