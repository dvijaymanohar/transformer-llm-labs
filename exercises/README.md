# Exercises

- Implement sinusoidal/RoPE position handling.
- Split QKV into heads by hand and verify shapes.
- Demonstrate why the causal mask prevents future-token access.
- Implement one transformer block without `nn.MultiheadAttention`.
- Add greedy autoregressive generation.
- Add a KV cache and measure decode-time/memory trade-offs.
- Sweep context length, batch, and precision and explain the bottleneck shift.
