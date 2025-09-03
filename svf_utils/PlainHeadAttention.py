import torch.nn as nn
import torch
import torch.nn.functional as F


class PlainMultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads=12,
            dropout=0.,
            bias=True,
            kdim=None,
            vdim=None,
            batch_first=False,
            adapt=False,
            prompt_length=0,
            prompt_dim=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            assert NotImplementedError
        else:

            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def init_weights(self):
        pass

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):
        
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape

        src_len, _, _ = key.shape

        E = query.size(-1)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None



    def set_parameters(self, torch_tgt_module):

        assert self.head_dim == torch_tgt_module.head_dim
        assert self.num_heads == torch_tgt_module.num_heads
        assert isinstance(torch_tgt_module, nn.MultiheadAttention)
        assert self.embed_dim == torch_tgt_module.embed_dim
        assert self.batch_first == torch_tgt_module.batch_first
        assert self.dropout == torch_tgt_module.dropout
        assert self.kdim == torch_tgt_module.kdim
        assert self.vdim == torch_tgt_module.vdim

        # Extract the existing weights and biases
        existing_weight = torch_tgt_module.in_proj_weight.data
        existing_bias = torch_tgt_module.in_proj_bias.data if torch_tgt_module.in_proj_bias is not None else None

        # Initialize proj
        self.proj.weight.data.copy_(torch_tgt_module.out_proj.weight.data)
        if self.proj.bias is not None:
            self.proj.bias.data.copy_(torch_tgt_module.out_proj.bias.data)

        # Initialize q_proj
        self.q_proj.weight.data.copy_(existing_weight[:self.embed_dim, :])
        if existing_bias is not None:
            self.q_proj.bias.data.copy_(existing_bias[:self.embed_dim])

        # Initialize k_proj
        self.k_proj.weight.data.copy_(existing_weight[self.embed_dim:2*self.embed_dim, :])
        if existing_bias is not None:
            self.k_proj.bias.data.copy_(existing_bias[self.embed_dim:2*self.embed_dim])

        # Initialize v_proj
        self.v_proj.weight.data.copy_(existing_weight[2*self.embed_dim:, :])
        if existing_bias is not None:
            self.v_proj.bias.data.copy_(existing_bias[2*self.embed_dim:])
        