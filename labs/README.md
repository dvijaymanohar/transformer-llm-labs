# Transformer labs

```bash
python examples/attention.py
python examples/rope.py
python examples/tiny_decoder.py
python examples/generation.py
python examples/kv_cache_attention.py
python benchmarks/prefill_decode.py
```

Exercises:
- add RoPE to TinyBlock Q/K tensors;
- implement a true decoder KV cache rather than the standalone attention cache demo;
- compare naive decode (recompute full prefix) against cached decode;
- measure cache memory as context and batch grow;
- profile prefill and decode separately.
