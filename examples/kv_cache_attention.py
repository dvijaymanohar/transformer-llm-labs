import math, torch

def step_attention(q,k_new,v_new,cache=None):
    # q,k_new,v_new: [B,H,1,D]
    if cache is None:
        k,v=k_new,v_new
    else:
        old_k,old_v=cache
        k=torch.cat([old_k,k_new],dim=-2)
        v=torch.cat([old_v,v_new],dim=-2)
    scores=q@k.transpose(-2,-1)/math.sqrt(q.size(-1))
    out=scores.softmax(-1)@v
    return out,(k,v)

if __name__=="__main__":
    torch.manual_seed(0)
    cache=None
    for step in range(4):
        q=torch.randn(1,2,1,8); k=torch.randn_like(q); v=torch.randn_like(q)
        out,cache=step_attention(q,k,v,cache)
        print("step",step,"cache_tokens",cache[0].size(-2),"out",tuple(out.shape))
