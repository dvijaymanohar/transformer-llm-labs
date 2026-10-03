# Benchmarks

Measure prefill and decode separately. Warm up, synchronize CUDA around host timers, repeat, and report median/spread. Record batch, prompt length, generated length, hidden size, head count, dtype, device, framework version, and memory footprint.
