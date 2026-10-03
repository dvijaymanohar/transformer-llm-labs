# Transformer & LLM Labs

Build transformer mechanisms from small, inspectable PyTorch examples before using high-level LLM frameworks.

## Sequence
embeddings → RoPE → QKV projections → scaled dot-product attention → causal masking → multi-head attention → MLP → residual/norm → transformer block → tiny decoder → autoregressive generation → KV cache.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/attention.py
python examples/tiny_decoder.py
pytest -q
```
