import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from train_masked_tfm import MaskedTFM, masked_loss


class TFMDataset(Dataset):
    def __init__(self, X_num, X_cat):

        self.X_num = X_num
        self.X_cat = X_cat

        if self.X_num.shape[0] != self.X_cat.shape[0]:
            raise ValueError("Mismatch between number of numeric and cateogrical rows")

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, key):
        return self.X_num[key, ...], self.X_cat[key, ...]


def train_one_epoch(model, train_dataloader, optimizer, device):

    model.train()
    loss_total, n_batches = 0, 0

    for x_num, x_cat in train_dataloader:
        # Dataloader yields CPU tensors by default
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)

        optimizer.zero_grad()  # reset gradients for each batch
        yhats, logits, mask = model(x_num, x_cat)  # forward pass
        loss = masked_loss(yhats, logits, mask, x_num, x_cat)
        loss.backward()  # calculate gradients
        optimizer.step()  # update parameters
        loss_total += loss.item()
        n_batches += 1

    avg_loss = loss_total / n_batches

    return avg_loss


@torch.no_grad()
def evaluate(model, val_dataloader, device):
    model.eval()
    loss_total, n_batches = 0, 0

    for x_num, x_cat in val_dataloader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)

        yhats, logits, mask = model(x_num, x_cat)
        loss = masked_loss(yhats, logits, mask, x_num, x_cat)
        loss_total += loss.item()
        n_batches += 1

    avg_loss = loss_total / n_batches
    return avg_loss


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

    # Define masked tabular foundation model
    masked_tfm = MaskedTFM(D, P_num, P_cat, cat_cardinalities, p_mask)
    masked_tfm.to(device)  # move parameters to device before optimizer instantiation

    # ------------   Training loop   ------------

    optimizer = torch.optim.Adam(masked_tfm.parameters(), lr=0.001)
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(
            masked_tfm, train_dataloader, optimizer, device=device
        )
        val_loss = evaluate(masked_tfm, val_dataloader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(masked_tfm.state_dict(), "masked_tfm.pt")

        print(
            f"Epoch {epoch + 1}/{n_epochs}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, best_val_loss = {best_val_loss}"
        )

    # Potential further inclusions:
    # - Learning rate scheduler (StepLR/Cosine/OneCycle); log dynamic lr each epoch
    # - Save latest checkpoint each epoch along with best
    # - Track richer metrics (e.g., numeric/categorical loss components) for train + val
    # - Gradient clipping for stability on harder training runs
    # - Mixed precision for faster GPU training
    # - Early stopping with patience on val metric
    # - Hyperparameters/paths into a config object/file
    # - Deterministic settings + seed logging for reproducible runs.
    # - Lightweight logger (CSV/TensorBoard/W&B) for curve visualization.

    # ------------   Load best model   ------------
    # Note that this is an inherently self-supervised task, so we omit any holdout set evaluation
    state = torch.load("masked_tfm.pt", map_location=device)
    masked_tfm.load_state_dict(state)
