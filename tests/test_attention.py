import torch
from examples.attention import causal_attention

def test_shapes_and_causality():
    torch.manual_seed(0)
    q=k=v=torch.randn(2,3,5,4)
    out,p=causal_attention(q,k,v)
    assert out.shape==(2,3,5,4)
    assert torch.triu(p,diagonal=1).abs().max().item()==0.0
    torch.testing.assert_close(p.sum(-1),torch.ones_like(p.sum(-1)))
