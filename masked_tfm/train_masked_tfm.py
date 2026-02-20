# Purpose: Turn existing forward.py and attention.py into a trainable PyTorch script
# At a high level, this involves:
# - Pulling learned tensors into an nn.Module
#     - nn.Parameter for numeric embeddings and numeric output head weights
#     - nn.Embedding for categorical lookups
#     - nn.Linear for categorical output heads
# - Building a training loop
#     - Create optimizer
#     - Sample a mask
#     - Forward pass + loss calculation
#     - zero_grad --> backward --> step

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.functional import cross_entropy
from torch.optim import Adam


def sample_batch(B, P_num, P_cat, cat_cardinalities, device):
    """Function to sample a batch of size B per training step."""
    # Generate numeric features
    age_mean, age_std = 25.0, 5.0
    weight_mean, weight_std = 170.0, 20.0
    height_mean, height_std = 65.0, 10.0
    means = torch.tensor([age_mean, weight_mean, height_mean], device=device)
    stds = torch.tensor([age_std, weight_std, height_std], device=device)
    X_num = torch.randn(B, P_num, device=device) * stds + means
    X_num_mean = X_num.mean(dim=0, keepdim=True)
    X_num_std = X_num.std(dim=0, keepdim=True, unbiased=False)
    X_num = (X_num - X_num_mean) / X_num_std
    # Generate categorical features
    X_cat = (torch.rand(B, P_cat, device=device) * cat_cardinalities).long()

    return X_num, X_cat


def masked_loss(yhats, logits, mask, X_num, X_cat):
    """
    Calculate masked loss after a forward pass

    :param yhats: Matrix of numeric feature predictions
    :param logits: List of P_cat logits
    :param mask: Mask of shape (B, P, 1) (whether feature p of batch element b is masked)
    :param X_num: Original numeric features (targets for numeric predictions)
    :param X_cat: Original categorical features (targets for class logits)
    """
    device = X_num.device
    dtype = X_num.dtype

    P_num = X_num.shape[-1]
    P_cat = X_cat.shape[-1]

    mask_num = mask[:, :P_num, 0].bool()
    mask_cat = mask[:, -P_cat:, 0].bool()

    yhat_masked = yhats[mask_num]
    y_true_masked = X_num[mask_num]

    # Guard against an empty mask
    weighted_mse = None
    if yhat_masked.numel() > 0:
        se = ((yhat_masked - y_true_masked) ** 2).sum()
        n_masked_num = yhat_masked.numel()
        weighted_mse = se / n_masked_num

    # Running totals for CE loss over masked elements
    ce_loss_sum = torch.zeros((), device=device, dtype=dtype)
    n_cat = torch.zeros((), device=device, dtype=torch.long)

    for i, decoded in enumerate(logits):
        mask_cat_cur = mask_cat[:, i]
        if mask_cat_cur.any():
            logits_masked = decoded[mask_cat_cur]
            y_true_masked = X_cat[:, i][mask_cat_cur]
            ce_loss = cross_entropy(logits_masked, y_true_masked, reduction="sum")
            ce_loss_sum += ce_loss
            n_cat += mask_cat_cur.sum()

    # Calculate total loss, guarding against empty masks
    if weighted_mse is None:
        weighted_mse = torch.zeros((), device=device, dtype=dtype)
    if n_cat.item() > 0:
        weighted_ce_loss = ce_loss_sum / n_cat
    else:
        weighted_ce_loss = torch.zeros((), device=device, dtype=dtype)
    loss_total = weighted_mse + weighted_ce_loss

    return loss_total


class MaskedTFM(nn.Module):
    def __init__(self, D, P_num, P_cat, cat_cardinalities, p_mask):
        """
        Constructor for class MaskedTFM (masked tabular foundation model).

        :param D: Embedding dimension
        :param P_num: Number of numeric features
        :param P_cat: Number of categorical features
        :param cat_cardinalities: List of number of categories per categorical feature
        :param p_mask: probability of feature masking
        """
        super().__init__()

        # Total number of features
        P_total = P_num + P_cat

        # Learned numeric embeddings
        self.embed_num = nn.Parameter(torch.randn(P_num, D))
        # Learned categorical embeddings (lookups)
        self.embed_cat = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=int(k), embedding_dim=D)
                for k in cat_cardinalities
            ]
        )
        # Column embedding and mask vectors
        self.col_emb = nn.Parameter(torch.randn(P_total, D))
        self.mask_emb = nn.Parameter(torch.randn(D))

        # Single transformer block
        encoder_layer = TransformerEncoderLayer(
            d_model=D, nhead=1, dim_feedforward=12, batch_first=True
        )
        self.encoder_layer = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=1, enable_nested_tensor=False
        )

        self.heads_num_weights = nn.Parameter(torch.randn(P_num, D))
        self.heads_num_biases = nn.Parameter(torch.randn(P_num))
        self.heads_cat = nn.ModuleList(
            [nn.Linear(in_features=D, out_features=int(k)) for k in cat_cardinalities]
        )

        # Store attributes for later use
        self.P_num = P_num
        self.P_total = P_total
        self.p_mask = p_mask

    def forward(self, X_num, X_cat):

        B = X_num.shape[0]
        device = X_num.device
        dtype = X_num.dtype

        # Tokenize (embed)
        # Numeric: scale the embedding vector for that feature by the feature value
        X_num_emb = (
            X_num.unsqueeze(-1) * self.embed_num
        )  # unsqueeze to make room for hidden dim
        # Categorical: embedding lookup via matrix multiplication
        X_cats = []
        for p_cat_idx in range(X_cat.shape[-1]):
            X_cat_cur = X_cat[:, p_cat_idx]
            X_cat_cur_emb = self.embed_cat[p_cat_idx](X_cat_cur)
            X_cats.append(X_cat_cur_emb)

        # Concatenate along feature dim + add column embedding
        X_cat_emb = torch.stack(X_cats, dim=1)
        X_emb = torch.cat([X_num_emb, X_cat_emb], dim=1) + self.col_emb

        # Sample a mask; add a singleton embedding dimension for broadcasting
        mask = (
            (torch.rand(B, self.P_total, device=device) < self.p_mask)
            .unsqueeze(-1)
            .to(dtype)
        )
        # Apply: zero the masked features and add the mask embedding to those features
        X_emb_masked = (X_emb * (1 - mask)) + (mask * self.mask_emb)
        # Encoder forward pass
        H = self.encoder_layer(X_emb_masked)

        # Decoder forward pass: linear heads
        yhats = (H[:, : self.P_num, :] * self.heads_num_weights).sum(
            dim=-1
        ) + self.heads_num_biases

        # Decoder forward pass: iterate for categorical features
        logits = []
        for p_cat_idx in range(X_cat.shape[-1]):
            p_global = self.P_num + p_cat_idx
            H_cat_cur = H[:, p_global, :]
            logits_cur = self.heads_cat[p_cat_idx](H_cat_cur)
            logits.append(logits_cur)

        # Return everything needed for loss calculation
        return yhats, logits, mask


# Training loop
def run_training(
    D, P_num, P_cat, cat_cardinalities, p_mask, batch_size, n_steps, device
):
    tfm = MaskedTFM(D, P_num, P_cat, cat_cardinalities, p_mask)
    tfm.to(device)
    tfm.train()
    optimizer = Adam(tfm.parameters(), lr=0.001)
    for step in range(n_steps):
        X_num_batch, X_cat_batch = sample_batch(
            batch_size, P_num, P_cat, cat_cardinalities, device
        )
        optimizer.zero_grad()
        yhats, logits, mask = tfm(X_num_batch, X_cat_batch)
        loss = masked_loss(yhats, logits, mask, X_num_batch, X_cat_batch)
        loss.backward()
        optimizer.step()
        print(f"Step {step}: loss = {loss.item():.4f}")


if __name__ == "__main__":

    torch.manual_seed(42)

    # Configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    embedding_dim = 6
    P_num = 3  # number of numeric features
    P_cat = 2  # number of categorical features
    p_mask = 0.2  # mask probability
    n_steps = 10  # number of training steps

    # Categorical feature configs
    N_cls_cat_1 = 2  # sex (M/F)
    N_cls_cat_2 = 3  # handedness (R/L/Amb.)
    cat_cardinalities = torch.tensor(
        [N_cls_cat_1, N_cls_cat_2], device=device
    )  # fixed order: sex, handedness

    run_training(
        D=embedding_dim,
        P_num=P_num,
        P_cat=P_cat,
        cat_cardinalities=cat_cardinalities,
        p_mask=p_mask,
        batch_size=batch_size,
        n_steps=n_steps,
        device=device,
    )
