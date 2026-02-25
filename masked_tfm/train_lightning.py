import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from train_masked_tfm import masked_loss
from train_torch import TFMDataset


# Init and forward pulled from train_masked_tfm.py
class LitMaskedTFM(L.LightningModule):
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
        self.save_hyperparameters()

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

    def training_step(self, batch, batch_idx):
        x_num, x_cat = batch
        yhats, logits, mask = self(x_num, x_cat)
        loss = masked_loss(yhats, logits, mask, x_num, x_cat)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num, x_cat = batch
        yhats, logits, mask = self(x_num, x_cat)
        loss = masked_loss(yhats, logits, mask, x_num, x_cat)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configs
    B = 32  # batch size
    D = 6  # embedding dim
    p_mask = 0.2  # mask probability
    n_epochs = 10

    # Generate the same mock data we have in other scripts
    N = 1000  # number of rows
    P_num = 3  # numeric features: age, weight, height
    age_mean, age_std = 25.0, 5.0
    weight_mean, weight_std = 170.0, 20.0
    height_mean, height_std = 65.0, 10.0
    means = torch.tensor([age_mean, weight_mean, height_mean])
    stds = torch.tensor([age_std, weight_std, height_std])
    X_num = torch.randn(N, P_num) * stds + means
    P_cat = 2  # categorical features: sex, handedness
    N_cls_cat_1 = 2  # M/F
    N_cls_cat_2 = 3  # R/L/Amb.
    cat_cardinalities = torch.tensor([N_cls_cat_1, N_cls_cat_2])
    X_cat = (torch.rand(N, P_cat) * cat_cardinalities).long()

    # Dataset
    tfm_dataset = TFMDataset(X_num, X_cat)

    # TVT split
    p_train, p_val = 0.8, 0.2
    n_train = int(p_train * N)
    n_val = N - n_train
    splits = random_split(tfm_dataset, lengths=[n_train, n_val])

    # Extract splits from base dataset
    X_num_train = X_num[splits[0].indices]
    X_cat_train = X_cat[splits[0].indices]
    X_num_val = X_num[splits[1].indices]
    X_cat_val = X_cat[splits[1].indices]

    # Standardize numeric features
    X_num_mean = X_num_train.mean(dim=0, keepdim=True)
    X_num_std = X_num_train.std(dim=0, keepdim=True, unbiased=False)
    X_num_train = (X_num_train - X_num_mean) / X_num_std
    X_num_val = (X_num_val - X_num_mean) / X_num_std

    # Train/val datasets + dataloaders (~ num_workers, pin_memory, etc)
    train_dataset = TFMDataset(X_num_train, X_cat_train)
    val_dataset = TFMDataset(X_num_val, X_cat_val)
    train_dataloader = DataLoader(train_dataset, batch_size=B, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=B, shuffle=False)

    # Train
    lit_masked_tfm = LitMaskedTFM(D, P_num, P_cat, cat_cardinalities, p_mask)
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        filename="masked_tfm_{epoch}_{val_loss:.4f}",
        auto_insert_metric_name=True,
    )
    trainer = L.Trainer(max_epochs=n_epochs, callbacks=[ckpt])
    trainer.fit(
        model=lit_masked_tfm,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Load best model (no evaluation per SSL)
    best_lit_masked_tfm = LitMaskedTFM.load_from_checkpoint(ckpt.best_model_path)
