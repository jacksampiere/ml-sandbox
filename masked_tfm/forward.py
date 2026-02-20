# Goal:
# - Build per-column tokens for a batch of tabular rows
# - Randomly mask some columns
# - Run a Transformer encoder over column-tokens
# - Predict original values for masked columns only
# - Compute masked reconstruction loss (MSE for numeric, CE for categorical)

# =========================
# 0) Imports + config
# =========================
# - Define: B (batch size), Pn (# numeric cols), Pc (# categorical cols), P=Pn+Pc, d (embed dim)
# - Define: cat_cardinalities: list of length Pc giving num categories K_j per categorical column

import torch
from torch.nn.functional import one_hot, cross_entropy
from torch.nn import TransformerEncoderLayer, TransformerEncoder

torch.manual_seed(42)
print()
print(
    "------------------------   TABULAR FOUNDATION MODEL MVP   ------------------------"
)
print()

# =========================
# CONSTANTS
# =========================
B = 32  # batch size
D = 6  # embedding dimension
P_MASK = 0.2  # mask probability
# Number of numeric features
P_num = 3
# Numeric features
age_mean, age_std = 25.0, 5.0
weight_mean, weight_std = 170.0, 20.0
height_mean, height_std = 65.0, 10.0
# Number of categorical features (with some other variables defined for later use)
N_cls_cat_1 = 2  # sex (M/F)
N_cls_cat_2 = 3  # handedness (R/L/Amb.)
cat_cardinalities = torch.tensor([N_cls_cat_1, N_cls_cat_2])
P_cat = cat_cardinalities.size()[0]
# Total number of features
P = P_num + P_cat

# =========================
# 1) Spoof data (B, P)
# =========================

# Normally distributed continuous variables
# age (years), height (cm), weight (kg)
means = torch.tensor([age_mean, weight_mean, height_mean])
stds = torch.tensor([age_std, weight_std, height_std])
X_num = torch.randn(B, P_num) * stds + means
# Standardize (reconciles numeric variable scales + ensures MSE doesn't squash CE loss)
X_num_mean = X_num.mean(dim=0, keepdim=True)
X_num_std = X_num.std(dim=0, keepdim=True, unbiased=False)
X_num = (X_num - X_num_mean) / X_num_std
# Create uniform distribution for each category
# Scale by number of available categories; truncate to integer class indices
X_cat = (torch.rand(B, P_cat) * cat_cardinalities).long()

print(f"Raw data has shape: {torch.cat([X_num, X_cat], axis=1).shape}")
print()

# Note: we don't create the full data matrix yet
# This is because we need to embed the numeric features via a projection
# In contrast, we embed the categorical features via an embedding lookup

# =========================
# 2) Build token embeddings per column --> (B, P, D)
# =========================

# Numeric embedding: for each numeric column j: map scalar -> vector in R^d

# First, we add a singleton dimension at the end index
# This will allow for expansion of the hidden dimensions
# i.e., get one embedding vector per feature (B, P_num, D)
X_num = X_num.unsqueeze(-1)
# Initialize matrix of embedding vectors
embed_num = torch.randn(P_num, D)  # ---------- LEARNED MATRIX ----------
# Embed numeric cols
# Embedding of numeric features is element-wise (broadcasted across batch dim)
# Hence, no torch.matmul
X_emb_num = X_num * embed_num
# Categorical embedding lookup: map each class to a unique embedding
# i.e., column with N classes --> (1, D) with N unique rows

# We need to run a lookup for each class
# This is a matmul by a lookup matrix of shape (num_classes, D)
# This means we have to one-hot encode each categorical feature
X_cat_1 = one_hot(X_cat[:, 0]).to(
    torch.float32
)  # need float version of ints to multiply by float lookups
embed_cat_1 = torch.randn(N_cls_cat_1, D)  # ---------- LEARNED MATRIX ----------
X_emb_cat_1 = torch.matmul(
    X_cat_1, embed_cat_1
)  # embedding of categorical features (lookup) is a matmul (--> (B, D))

X_cat_2 = one_hot(X_cat[:, 1]).to(torch.float32)
embed_cat_2 = torch.randn(N_cls_cat_2, D)  # ---------- LEARNED MATRIX ----------
X_emb_cat_2 = torch.matmul(X_cat_2, embed_cat_2)

# Concatenate tokens in a fixed order (e.g., all numeric then all categorical):
# - X: (B, P, d)
# Note we add a singleton dimension in middle of categorical embeddings (one feature)
X_emb = torch.cat(
    [X_emb_num, X_emb_cat_1.unsqueeze(1), X_emb_cat_2.unsqueeze(1)], dim=1
)
# Define these for later based on how we created and concatenated the data above
age_idx, weight_idx, height_idx = 0, 1, 2
sex_idx, hand_idx = 3, 4

# Column embeddings: (P, d) broadcasted to (B, P, d)

# Think of a column embedding as a numeric proxy for column label
col_emb = torch.randn(P, D)  # ---------- LEARNED MATRIX ----------
X_emb = X_emb + col_emb

print(f"Embedded feature matrix has shape: {X_emb.shape}")
print(
    "i.e., 32 rows in the table, 5 features, and each feature is embedded to a vector of 6 elements"
)
print()

# =========================
# 3) Sample a mask and apply it
# =========================
# - Sample M: (B, P) boolean mask (mask probability pm)
# - Create a learned mask_token: (d,)
# - Replace X[b, j, :] with mask_token where M[b, j] == True

# The masking process:

# (1) The data
# Embedded features of shape (B, P, D)
# Create a binary mask indicator of shape (B, P, 1) ("which features should we mask for each batch element?")

# (2) Incorporating the mask vector
# Initialize a mask vector of shape (1, 1, D)
# For any batch item i with masked feature embedding j, we replace the mask with the mask vector (via summation)

# Pseudo-code:
# for i in range(B):
#     for j in range(P):
#         if mask_indicator[i, j,...].item(): # True or False
#             X_emb[i, j, :] = mask_vec


# Create the binary mask
# Feature j of batch item i is masked if M[i, j,...] = 1 else unmasked
mask = (
    (torch.rand(B, P) < P_MASK).to(torch.float32).unsqueeze(-1)
)  # singleton so we can broadcast across batch dim
# Initialize the mask vector (length D)
# Reshape to enable multiplication by X_emb of shape (B, P, D)
mask_vec = torch.randn(D).view(1, 1, -1)  # ---------- LEARNED VECTOR ----------
# Apply to mask to (1) the embeddings and (2) the mask vector, then sum
X_emb_masked = X_emb * (1 - mask) + mask * mask_vec

print(f"Masked embedded feature matrix has shape: {X_emb_masked.shape}")
print()

# =========================
# 4) Transformer encoder forward
# =========================
# - Feed X_masked into TransformerEncoder
# - Be mindful of expected shape: many PyTorch modules want (B, P, D), so you may transpose
#
# Output:
# - H: (B, P, d) final hidden states per token
encoder_layer = TransformerEncoderLayer(
    d_model=D, nhead=1, dim_feedforward=12, batch_first=True
)
encoder = TransformerEncoder(
    encoder_layer=encoder_layer, num_layers=1, enable_nested_tensor=False
)  # odd n_heads disables nested tensor
H = encoder(X_emb_masked)
# Save for later use (e.g., attention from scratch)
torch.save(X_emb_masked, "X_emb_masked.pt")
print(f"Output of TransformerEncoder has shape {H.shape}")
print()

# =========================
# 5) Decode predictions for each column
# =========================

# Numeric columns:
# - For token positions 0..Pn-1, apply linear head d->1
# - yhat_num: (B, Pn) predicted numeric values

# Consider a standalone forward pass for one feature (age):
W_age = torch.randn(D)  # weight vector
b_age = torch.randn(1)  # scalar bias
yhat_age = torch.matmul(H[:, age_idx, ...], W_age) + b_age

# Rather than doing this for each feature separately, reconcile into a single matrix:
W_weight = torch.randn(D)  # weight vector
b_weight = torch.randn(1)  # scalar bias
W_height = torch.randn(D)  # weight vector
b_height = torch.randn(1)  # scalar bias
# Data
H_num = H[:, :P_num, ...]
# Full weight matrix
W_num = torch.stack([W_age, W_weight, W_height])
# Full bias vector
b_num = torch.stack([b_age, b_weight, b_height]).squeeze(-1)
H_num = H[:, :P_num, ...]
W_num = torch.stack([W_age, W_weight, W_height])
b_num = torch.stack([b_age, b_weight, b_height]).squeeze(-1)
yhat_num = (H_num * W_num).sum(dim=-1) + b_num  # broadcast dot products
# Check equality of standalone vs. matrix version of linear head forward pass
yhat_weight = torch.matmul(H[:, weight_idx, ...], W_weight) + b_weight
yhat_height = torch.matmul(H[:, height_idx, ...], W_height) + b_height
yhat_num_expected = torch.stack([yhat_age, yhat_weight, yhat_height], axis=1)
assert torch.allclose(yhat_num, yhat_num_expected, atol=1e-4)

# Categorical columns:
# - For each categorical column j, apply linear head d->K_j to its token position
# - logits_cat_j: (B, K_j)

# Sex (categorical variable 1)
W_sex = torch.randn(D, N_cls_cat_1)
b_sex = torch.randn(N_cls_cat_1)
logits_sex = (
    torch.matmul(H[:, sex_idx, ...], W_sex) + b_sex
)  # no need for softmax, just calculating loss via logits

# Handedness (categorical variable 2)
W_hand = torch.randn(D, N_cls_cat_2)
b_hand = torch.randn(N_cls_cat_2)
logits_hand = (
    torch.matmul(H[:, hand_idx, ...], W_hand) + b_hand
)  # no need for softmax, just calculating loss via logits

print(f"Shape of linear head outputs of numeric features: {yhat_num.shape}")
print(f"Shape of linear head outputs (sex; n_cls = 2): {logits_sex.shape}")
print(f"Shape of linear head outputs (handedness; n_cls = 3): {logits_hand.shape}")
print()

# Note: reconciling categorical features with different numbers of classes is messy
# So, we leave the forward passes separate here
# In practice (i.e., when training), prefer keeping a `ModuleList` of columns and looping

# =========================
# 6) Compute masked reconstruction loss
# =========================

# Numeric masked loss:
# - Identify masked numeric positions: M_num = M[:, :Pn] -> (B, Pn)
# - Compare yhat_num to x_num only where M_num is True
# - loss_num = mean squared error over masked numeric entries
mask_num = mask[:, :P_num, ...].squeeze(-1).to(torch.bool)
yhat_num_masked = yhat_num[mask_num]  # full-matrix forward pass output from above
y_num = X_num.squeeze(-1)  # true feature values
y_num_masked = y_num[mask_num]
L_num = torch.mean((yhat_num_masked - y_num_masked) ** 2)  # MSE
print(f"Number of masked numeric elements: {mask_num.sum()}")
print(f"Number of elements for MSE calculation: {yhat_num_masked.shape[0]}")
print(f"MSE across numeric features: {L_num.item():.4f}")
print()

# Categorical masked loss:
# - Identify masked cat positions: M_cat = M[:, Pn:] -> (B, Pc)
# - For each categorical column j:
#   - only include rows b where M_cat[b, j] is True
#   - CE(logits_cat_j[b], x_cat[b, j])
# - loss_cat = mean CE over masked categorical entries

# Retrieve mask
mask_sex = mask[:, sex_idx, ...].squeeze(-1).to(torch.bool)
# Ground truth values in original data
y_sex = X_cat[:, 0, ...]  # sex is at index 0 when constructing `X_cat`
# Compute masked loss
logits_sex_masked = logits_sex[mask_sex]
y_sex_masked = y_sex[mask_sex]
# We used a 2-logit head --> CE instead of BCE
L_sex = cross_entropy(logits_sex_masked, y_sex_masked)  # empty mask guard omitted per seed
print(f"Number of masked categorical elements (sex): {mask_sex.sum()}")
print(f"Number of elements for CE loss calculation (sex): {y_sex_masked.shape[0]}")
print(f"CE loss (sex): {L_sex:.4f}")
print()

# Same thing but for handedness features
mask_hand = mask[:, hand_idx, ...].squeeze(-1).to(torch.bool)
y_hand = X_cat[:, 1, ...]
logits_hand_masked = logits_hand[mask_hand]
y_hand_masked = y_hand[mask_hand]
L_hand = cross_entropy(logits_hand_masked, y_hand_masked)  # empty mask guard omitted per seed
print(f"Number of masked categorical elements (handedness): {mask_hand.sum()}")
print(
    f"Number of elements for CE loss calculation (handedness): {y_hand_masked.shape[0]}"
)
print(f"CE loss (handedness): {L_hand:.4f}")
print()

# ---------   Check CE loss implementation from scratch   ---------
logits_check = logits_hand_masked
y_check = y_hand_masked
# Subtract the max logits from all logits of each sample (stability)
L_shift = logits_check - logits_check.max(axis=1).values.unsqueeze(-1)
# (1) Exponentiate the shifted logits
# (2) Sum across classes (get one sum of logits per batch item)
# (3) Take log
# Note: this is the denominator of softmax
log_z = torch.log(
    torch.sum(torch.exp(L_shift), axis=1),
)
# Get the shifted logit corresponding to the correct class
# Note: since we shifted by the max, this value is zero for correct predictions
row_idx = torch.arange(y_check.shape[0])
L_true = L_shift[row_idx, y_check]  # think of y_check as class_idx
# Subtract softmax denom from shifted true logits
# This converts the logit into the log‑probability of the true class
L_per_row = -(L_true - log_z)
L_check = L_per_row.mean()
assert torch.allclose(L_hand, L_check, atol=1e-4)
# -----------------------------------------------------------------

# Total categorical loss
L_cat = L_sex + L_hand
print(f"Total categorical CE loss: {L_cat:.4f}")
print()

# Total loss
L = L_num + L_cat
print(f"Total loss: {L:.4f}")

print()
print(
    "----------------------------------------------------------------------------------"
)
print()

# Next steps:
# - Include type embedding
# - Use distributional numeric loss instead of MSE
# - Use real data
