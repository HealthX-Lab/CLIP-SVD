import torch.nn as nn
import torch
import torch.nn.functional as F

class BiomedCLIPMultiHeadAttention(nn.Module):

    def __init__(
            self,
            dim: int = 768,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def set_parameters(self, torch_tgt_module):

        existing_weight = torch_tgt_module.qkv.weight.data
        existing_bias = torch_tgt_module.qkv.bias.data

        self.proj.weight.data.copy_(torch_tgt_module.proj.weight.data)
        self.proj.bias.data.copy_(torch_tgt_module.proj.bias.data)

        # Initialize q_proj
        self.q_proj.weight.data.copy_(existing_weight[:self.dim, :])
        self.q_proj.bias.data.copy_(existing_bias[:self.dim])

        # Initialize k_proj
        self.k_proj.weight.data.copy_(existing_weight[self.dim:2*self.dim, :])
        self.k_proj.bias.data.copy_(existing_bias[self.dim:2*self.dim])

        # Initialize v_proj
        self.v_proj.weight.data.copy_(existing_weight[2*self.dim:, :])
        self.v_proj.bias.data.copy_(existing_bias[2*self.dim:])
        