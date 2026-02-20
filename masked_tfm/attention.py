import torch
from torch.nn.functional import softmax
from torch.nn import MultiheadAttention

D = 6
torch.manual_seed(42)

# Read masked + embedded features from manual forward pass (see forward.py)
X_emb_masked = torch.load("X_emb_masked.pt")

# Instantiate attention block
mha = MultiheadAttention(embed_dim=D, num_heads=1, dropout=0, batch_first=True)
# Deactivate dropout layers + freeze batch norm layers
mha.eval()
# Extract params
W, biases = mha.in_proj_weight, mha.in_proj_bias
# Weight and biases are stacked into a single matrix/vector
W_q, W_k, W_v = W.chunk(3)
b_q, b_k, b_v = biases.chunk(3)
W_o, b_o = mha.out_proj.weight, mha.out_proj.bias

# Implement attention by hand
Q = X_emb_masked @ W_q.T + b_q
K = X_emb_masked @ W_k.T + b_k
V = X_emb_masked @ W_v.T + b_v
d_k = torch.tensor(D)  # embedding dim / num_heads
attn_inner = (Q @ K.mT) / torch.sqrt(d_k)  # .mT flips only the matrix dims, not batch
attn = softmax(attn_inner, dim=-1) @ V
attn_proj = attn @ W_o.T + b_o

# Compare to forward pass
attn_expected, _ = mha(
    query=X_emb_masked, key=X_emb_masked, value=X_emb_masked
)  # self-attention --> same X to Q + K + V
assert torch.allclose(attn_proj, attn_expected, atol=1e-4)
