import torch
from torch import nn
import torch.nn.functional as F

VOCAB = 8
SEQ_LEN = 16
D_MODEL = 16
N_HEADS = 2
FFN_DIM = 64
DEPTH = 4
METHODS = ("untied", "albert_tied", "attention_shared", "ffn_shared", "tied_scalar", "tied_mirror")


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, N_HEADS, c // N_HEADS).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) * ((c // N_HEADS) ** -0.5)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        y = torch.softmax(scores, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_MODEL, FFN_DIM), nn.GELU(), nn.Linear(FFN_DIM, D_MODEL))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyAlbertLM(nn.Module):
    def __init__(self, method):
        super().__init__()
        assert method in METHODS
        self.method = method
        self.token_embedding = nn.Embedding(VOCAB, D_MODEL)
        self.position_embedding = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
        self.final_norm = nn.LayerNorm(D_MODEL)
        if method in ("albert_tied", "tied_scalar", "tied_mirror"):
            self.shared = Block()
            if method == "tied_scalar":
                self.depth_codes = nn.Parameter(torch.zeros(DEPTH))
            elif method == "tied_mirror":
                self.depth_codes = nn.Parameter(torch.zeros(DEPTH))
        elif method == "attention_shared":
            self.shared_attention = CausalAttention()
            self.layer_norm1 = nn.ModuleList(nn.LayerNorm(D_MODEL) for _ in range(DEPTH))
            self.layer_norm2 = nn.ModuleList(nn.LayerNorm(D_MODEL) for _ in range(DEPTH))
            self.layer_ffn = nn.ModuleList(FeedForward() for _ in range(DEPTH))
        elif method == "ffn_shared":
            self.layer_attention = nn.ModuleList(CausalAttention() for _ in range(DEPTH))
            self.layer_norm1 = nn.ModuleList(nn.LayerNorm(D_MODEL) for _ in range(DEPTH))
            self.layer_norm2 = nn.ModuleList(nn.LayerNorm(D_MODEL) for _ in range(DEPTH))
            self.shared_ffn = FeedForward()
        else:
            self.layers = nn.ModuleList(Block() for _ in range(DEPTH))

    def block_step(self, x, layer):
        if self.method in ("albert_tied", "tied_scalar", "tied_mirror"):
            y = self.shared(x)
            if self.method == "tied_scalar":
                delta = y - x
                y = x + (1.0 + self.depth_codes[layer]) * delta
            elif self.method == "tied_mirror":
                c, s = torch.cos(self.depth_codes[layer]), torch.sin(self.depth_codes[layer])
                even, odd = y[..., 0::2], y[..., 1::2]
                y = torch.stack((c * even - s * odd, s * even + c * odd), dim=-1).flatten(-2)
            return y
        if self.method == "attention_shared":
            x = x + self.shared_attention(self.layer_norm1[layer](x))
            return x + self.layer_ffn[layer](self.layer_norm2[layer](x))
        if self.method == "ffn_shared":
            x = x + self.layer_attention[layer](self.layer_norm1[layer](x))
            return x + self.shared_ffn(self.layer_norm2[layer](x))
        return self.layers[layer](x)

    def forward_prefixes(self, tokens):
        x = self.token_embedding(tokens) + self.position_embedding[:, :tokens.shape[1], :]
        result = []
        for layer in range(DEPTH):
            x = self.block_step(x, layer)
            h = self.final_norm(x)
            result.append(F.linear(h, self.token_embedding.weight))
        return result

    def forward_to_depth(self, tokens, depth):
        x = self.token_embedding(tokens) + self.position_embedding[:, :tokens.shape[1], :]
        for layer in range(depth):
            x = self.block_step(x, layer)
        return F.linear(self.final_norm(x), self.token_embedding.weight)

    def forward(self, tokens):
        return self.forward_to_depth(tokens, DEPTH)
