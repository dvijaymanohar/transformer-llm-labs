import torch

def rotate_half(x):
    x1=x[...,::2]; x2=x[...,1::2]
    return torch.stack((-x2,x1),dim=-1).flatten(-2)

def apply_rope(x):
    # x: [B,H,T,D], even D
    d=x.size(-1); t=x.size(-2)
    inv=1.0/(10000**(torch.arange(0,d,2,device=x.device,dtype=x.dtype)/d))
    pos=torch.arange(t,device=x.device,dtype=x.dtype)
    ang=torch.outer(pos,inv)
    cos=torch.repeat_interleave(ang.cos(),2,dim=-1)[None,None]
    sin=torch.repeat_interleave(ang.sin(),2,dim=-1)[None,None]
    return x*cos+rotate_half(x)*sin

if __name__=="__main__":
    torch.manual_seed(0)
    x=torch.randn(2,4,8,16)
    y=apply_rope(x)
    torch.testing.assert_close(x.norm(dim=-1),y.norm(dim=-1),rtol=1e-5,atol=1e-5)
    print("PASS",y.shape)
