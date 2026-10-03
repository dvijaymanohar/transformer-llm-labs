import math, torch

def causal_attention(q,k,v):
    # q,k,v: [B,H,T,D]
    scores=q @ k.transpose(-2,-1) / math.sqrt(q.size(-1))
    t=q.size(-2)
    mask=torch.triu(torch.ones(t,t,dtype=torch.bool,device=q.device),diagonal=1)
    scores=scores.masked_fill(mask,float("-inf"))
    probs=scores.softmax(dim=-1)
    return probs @ v, probs

if __name__=="__main__":
    torch.manual_seed(0)
    q=torch.randn(1,2,4,8); k=torch.randn(1,2,4,8); v=torch.randn(1,2,4,8)
    out,p=causal_attention(q,k,v)
    print("out",out.shape,"probs",p.shape)
    print("future_probability_mass", torch.triu(p,diagonal=1).sum().item())
