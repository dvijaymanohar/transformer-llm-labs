import torch
from tiny_decoder import TinyDecoder

@torch.inference_mode()
def greedy_generate(model,tokens,max_new_tokens=16):
    out=tokens
    for _ in range(max_new_tokens):
        logits=model(out)
        nxt=logits[:,-1].argmax(-1,keepdim=True)
        out=torch.cat([out,nxt],dim=1)
    return out

if __name__=="__main__":
    torch.manual_seed(0)
    m=TinyDecoder(vocab=128,d=64).eval()
    prompt=torch.tensor([[1,2,3,4]])
    result=greedy_generate(m,prompt,8)
    print(result.tolist())
