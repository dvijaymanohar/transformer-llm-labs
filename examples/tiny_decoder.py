import torch
from torch import nn
from attention import causal_attention

class TinyBlock(nn.Module):
    def __init__(self,d=64,h=4):
        super().__init__(); self.h=h; self.d=d
        self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
        self.norm1=nn.LayerNorm(d); self.norm2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x):
        b,t,d=x.shape; y=self.norm1(x); q,k,v=self.qkv(y).chunk(3,dim=-1)
        def heads(z): return z.view(b,t,self.h,d//self.h).transpose(1,2)
        a,_=causal_attention(heads(q),heads(k),heads(v))
        a=a.transpose(1,2).contiguous().view(b,t,d)
        x=x+self.proj(a); return x+self.mlp(self.norm2(x))

class TinyDecoder(nn.Module):
    def __init__(self,vocab=128,d=64):
        super().__init__(); self.emb=nn.Embedding(vocab,d); self.block=TinyBlock(d); self.lm=nn.Linear(d,vocab,bias=False)
    def forward(self,tok): return self.lm(self.block(self.emb(tok)))

if __name__=="__main__":
    torch.manual_seed(0); m=TinyDecoder(); x=torch.randint(0,128,(2,16)); y=m(x)
    print("logits",tuple(y.shape))
