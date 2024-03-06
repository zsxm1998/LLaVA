# Add by ZSXM
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, attn_drop=0., out_drop=0., bias=True, qdim=None, kdim=None, vdim=None, odim=None, o_proj=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, f"embed_dim must be divisible by num_heads, but got {embed_dim} and {num_heads}"
        self.scaler = self.head_dim ** -0.5
        
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.odim = odim if odim is not None else embed_dim

        self.q_proj = nn.Linear(self.qdim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, self.odim, bias=bias) if o_proj or self.odim!=embed_dim else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.o_drop = nn.Dropout(out_drop)

    def forward(self, query, key, value,
                need_weights=False,
                average_attn_weights=False,
                key_padding_mask=None,
                attn_mask=None):
        assert key.shape[:-1] == value.shape[:-1], \
            f"key's batch and sequence dims {key.shape[:-1]} do not match value's {value.shape[:-1]}"
        
        # compute in-projection
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value) #[B,N,C]

        # reshape q, k, v for multihead attention
        B, NQ, C = q.shape
        _, NK, _ = k.shape
        q = q.reshape(B, NQ, self.num_heads, self.head_dim).transpose(1, 2) #[B,H,NQ,HC]
        k = k.reshape(B, NK, self.num_heads, self.head_dim).transpose(1, 2) #[B,H,NK,HC]
        v = v.reshape(B, NK, self.num_heads, self.head_dim).transpose(1, 2) #[B,H,NK,HC]

        # process key_padding_mask and attn_mask
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool or torch.is_floating_point(key_padding_mask), \
                f"only bool and floating types of key_padding_mask are supported, but got {key_padding_mask.dtype}"
            assert key_padding_mask.shape == (B,NK), f"expecting key_padding_mask shape of {(B,NK)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(B,1,1,NK).expand(-1,self.num_heads,NQ,-1)
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = torch.masked_fill(torch.zeros_like(key_padding_mask), key_padding_mask, float('-inf'))
            if attn_mask is None:
                attn_mask = key_padding_mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool or torch.is_floating_point(attn_mask), \
                f"only bool and floating types of attn_mask are supported, but got {attn_mask.dtype}"
            if attn_mask.ndim == 2:
                assert attn_mask.shape == (NQ, NK), f"The shape of the 2D attn_mask should be {(NQ,NK)}, but got {attn_mask.shape}."
                attn_mask = attn_mask.view(1,1,NQ,NK).expand(B,self.num_heads,-1,-1)
            elif attn_mask.ndim == 4:
                assert attn_mask.shape == (B, self.num_heads, NQ, NK), f"The shape of the 4D attn_mask should be {(B,self.num_heads,NQ,NK)}, but got {attn_mask.shape}."
            else:
                raise RuntimeError(f"attn_mask's dimension should be 2 or 4, but got {attn_mask.ndim}")
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.masked_fill(torch.zeros_like(attn_mask), attn_mask, float('-inf'))
            if key_padding_mask is not None:
                attn_mask += key_padding_mask

        # compute attention
        attn = q @ k.transpose(-2,-1) * self.scaler #[B,H,NQ,NK]
        if attn_mask is not None:
            attn = attn + attn_mask.to(attn)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn) #[B,H,NQ,NK]
        # [B,H,NQ,NK] @ [B,H,NK,HC] -> [B,H,NQ,HC]
        attn_output = attn @ v

        # reshape attention output and apply out-projection
        attn_output = attn_output.transpose(1,2).reshape(B, NQ, C)
        attn_output = self.o_proj(attn_output)
        attn_output = self.o_drop(attn_output)

        if need_weights:
            # optionally average attention weights over heads
            if average_attn_weights:
                attn = attn.mean(dim=1) # [B,NQ,NK]
            return attn_output, attn
        else:
            return attn_output, None
        

class MultiheadSelfAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads=8, attn_drop=0., out_drop=0., bias=True, fea_dim=None, odim=None, o_proj=False):
        super().__init__(embed_dim, num_heads, attn_drop, out_drop, bias, fea_dim, fea_dim, fea_dim, odim or fea_dim, o_proj)

    def forward(self, x, need_weights=False, average_attn_weights=False, key_padding_mask=None, attn_mask=None):
        return super().forward(x, x, x,
                               need_weights=need_weights,
                               average_attn_weights=average_attn_weights,
                               key_padding_mask=key_padding_mask,
                               attn_mask=attn_mask)


class MultiheadCrossAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads=8, attn_drop=0., out_drop=0., bias=True, qdim=None, kvdim=None, odim=None, o_proj=False):
        super().__init__(embed_dim, num_heads, attn_drop, out_drop, bias, qdim, kvdim, kvdim, odim, o_proj)

    def forward(self, tgt, memory, need_weights=False, average_attn_weights=False, key_padding_mask=None, attn_mask=None):
        return super().forward(tgt, memory, memory,
                               need_weights=need_weights,
                               average_attn_weights=average_attn_weights,
                               key_padding_mask=key_padding_mask,
                               attn_mask=attn_mask)
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, activation=nn.GELU, dropout=0.):
        super(MLP, self).__init__()
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.drop = nn.Dropout(dropout)
        
        if isinstance(activation, str):
            if activation == "relu":
                self.act = nn.GELU()
            elif activation == "gelu":
                self.act = nn.ReLU()
            else:
                raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
        else:
            self.act = activation()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class QFormerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=16):
        super().__init__()
        self.mhca = MultiheadCrossAttention(embed_dim=out_dim, num_heads=num_heads, bias=False, kvdim=in_dim)
        self.mhsa = MultiheadSelfAttention(out_dim, num_heads, bias=False)
        self.mlp = MLP(out_dim, int(out_dim * 2))
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, query, image, need_weights=False, average_attn_weights=False):
        assert query.ndim == 3 and image.ndim == 3, f'query.ndim and image.ndim should be 3(B,N,C), but got {query.ndim} and {image.ndim}'
        out, attn_c = self.mhca(query, image, need_weights=need_weights, average_attn_weights=average_attn_weights)
        query = self.norm1(query + out)
        out, attn_s = self.mhsa(query, need_weights=need_weights, average_attn_weights=average_attn_weights)
        query = self.norm2(query + out)
        query = self.norm3(query + self.mlp(query))
        return (query, attn_c, attn_s) if need_weights else query


class QFormer(nn.Module):
    def __init__(self, num_query, num_layer, in_dim, out_dim):
        super().__init__()
        self.query = nn.Parameter(torch.empty(num_query, out_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.layers = nn.ModuleList([
            QFormerBlock(in_dim, out_dim) for _ in range(num_layer)
        ])

    def forward(self, image):
        assert image.ndim in [3, 4], f'Wrong image tensor shape: {image.shape}'
        query = self.query.expand(image.shape[-3], self.query.shape[0], self.query.shape[1])
        for i, layer in enumerate(self.layers):
            query = layer(query, image if image.ndim == 3 else image[i])
        return query