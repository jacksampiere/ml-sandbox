# Masked Tabular Transformer

This folder explores building a small masked tabular foundation model from first principles.  

## Script overview

- `forward.py`  
  A single forward pass on synthetic tabular data. It creates numeric and categorical token embeddings, applies random masking, runs a transformer encoder, and decodes masked targets with numeric and categorical heads.

- `attention.py`  
  A verification script that manually implements single-head self-attention and compares against `torch.nn.MultiheadAttention` using saved masked embeddings (`X_emb_masked.pt`) from `forward.py`.

- `train_masked_tfm.py`  
  A low-level prototype that defines a `MaskedTFM` module and custom `masked_loss` function and trains directly on sampled synthetic batches.

- `train_torch.py`  
   End-to-end PyTorch training pipeline that wraps the same core model/loss from  with a dataset, dataLoader, train/val/test splits, feature standardization, and epoch-based optimization.

## Running the code
Install uv:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Environment setup:
```shell
uv venv .venv
uv sync
source .venv/bin/activate
```

Running the scripts:
```shell
python <script_name>
```
