# Prefill, decode, and KV cache

**Prefill** processes the prompt and is relatively parallel across tokens. **Decode** generates tokens sequentially and often has a different latency/utilization regime.

A KV cache stores past key/value tensors so decode does not recompute attention states for the entire prefix.

## Required experiments
- sweep prompt length and measure prefill latency
- sweep generated-token count and measure per-token decode latency
- compare naive recomputation with KV-cache reuse
- sweep batch size and precision
- record memory growth with context length and batch size

Report TTFT-like prefill latency and inter-token-like decode timing separately.
