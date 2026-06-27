# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if is_torch_npu_available(check_device=False):
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            # Fallback to pure PyTorch/einops implementation if flash_attn is not available or fails to import
            import torch
            import torch.nn.functional as F
            from einops import rearrange as einops_rearrange

            class IndexFirstAxis(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input, indices):
                    ctx.save_for_backward(indices)
                    assert input.ndim >= 2
                    ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
                    second_dim = other_shape.numel()
                    flat_input = input.flatten(1)
                    gathered = torch.gather(
                        flat_input, 0, indices.unsqueeze(-1).expand(-1, second_dim)
                    )
                    return gathered.reshape(-1, *other_shape)

                @staticmethod
                def backward(ctx, grad_output):
                    (indices,) = ctx.saved_tensors
                    assert grad_output.ndim >= 2
                    other_shape = grad_output.shape[1:]
                    flat_grad_output = grad_output.flatten(1)
                    grad_input = torch.zeros(
                        [ctx.first_axis_dim, flat_grad_output.shape[1]],
                        device=grad_output.device,
                        dtype=grad_output.dtype,
                    )
                    grad_input.scatter_(0, indices.unsqueeze(-1).expand(-1, flat_grad_output.shape[1]), flat_grad_output)
                    return grad_input.reshape(ctx.first_axis_dim, *other_shape), None

            fallback_index_first_axis = IndexFirstAxis.apply


            class IndexPutFirstAxis(torch.autograd.Function):
                @staticmethod
                def forward(ctx, values, indices, first_axis_dim):
                    ctx.save_for_backward(indices)
                    assert indices.ndim == 1
                    assert values.ndim >= 2
                    output = torch.zeros(
                        first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
                    )
                    output[indices] = values
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    (indices,) = ctx.saved_tensors
                    grad_values = grad_output[indices]
                    return grad_values, None, None

            fallback_index_put_first_axis = IndexPutFirstAxis.apply


            def fallback_unpad_input(hidden_states, attention_mask):
                """
                Arguments:
                    hidden_states: (batch, seqlen, ...)
                    attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
                """
                seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
                indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
                max_seqlen_in_batch = seqlens_in_batch.max().item()
                cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
                return (
                    fallback_index_first_axis(hidden_states.flatten(0, 1), indices),
                    indices,
                    cu_seqlens,
                    max_seqlen_in_batch,
                )


            def fallback_pad_input(hidden_states, indices, batch, seqlen):
                """
                Arguments:
                    hidden_states: (total_nnz, ...), the flattened hidden states.
                    indices: (total_nnz), the indices of masked tokens.
                    batch: int, batch size for the padded sequence.
                    seqlen: int, maximum sequence length for the padded sequence.
                """
                output = fallback_index_put_first_axis(hidden_states, indices, batch * seqlen)
                return output.reshape(batch, seqlen, *hidden_states.shape[1:])

            index_first_axis = fallback_index_first_axis
            pad_input = fallback_pad_input
            rearrange = einops_rearrange
            unpad_input = fallback_unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.index_first_axis`
      - On NPU: `transformers.integrations.npu_flash_attention.index_first_axis`
        (falls back to `transformers.modeling_flash_attention_utils._index_first_axis`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.pad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.pad_input`
        (falls back to `transformers.modeling_flash_attention_utils._pad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.rearrange`
      - On NPU: `transformers.integrations.npu_flash_attention.rearrange`
        (falls back to `einops.rearrange` if no dedicated NPU implementation exists).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.unpad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.unpad_input`
        (falls back to `transformers.modeling_flash_attention_utils._unpad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
