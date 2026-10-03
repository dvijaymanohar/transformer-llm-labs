import statistics,time,torch
from examples.tiny_decoder import TinyDecoder
from examples.generation import greedy_generate

device="cuda" if torch.cuda.is_available() else "cpu"
m=TinyDecoder(vocab=128,d=64).to(device).eval()

def sync():
    if device=="cuda": torch.cuda.synchronize()

for t in [16,64,256]:
    x=torch.randint(0,128,(1,t),device=device)
    for _ in range(3):m(x)
    sync(); samples=[]
    for _ in range(10):
        a=time.perf_counter();m(x);sync();samples.append((time.perf_counter()-a)*1000)
    print({"phase":"prefill","tokens":t,"median_ms":statistics.median(samples)})
for prompt in [16,64]:
    x=torch.randint(0,128,(1,prompt),device=device)
    sync();a=time.perf_counter();greedy_generate(m,x,16);sync()
    print({"phase":"naive_decode","prompt":prompt,"new_tokens":16,"total_ms":(time.perf_counter()-a)*1000})
