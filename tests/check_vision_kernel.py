"""GB10 numerical checks against the MiMo checkpoint attention definition."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as N
import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--original-kernel', type=Path,
                    default=Path('/opt/mimo26/original_prefill_attention.py'))
parser.add_argument('--output', type=Path)
args = parser.parse_args()
spec=importlib.util.spec_from_file_location('original_prefill', args.original_kernel)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
torch.manual_seed(73419)
assert torch.cuda.get_device_capability()==(12,1)
rows=[]

def oracle(q,k,v,lengths,window,sinks,mode='key0',causal=False):
    outputs=[];start=0
    for n in lengths:
        a=q[start:start+n].float().transpose(0,1)
        b=k[start:start+n].float().repeat_interleave(4,dim=1).transpose(0,1)
        c=v[start:start+n].float().repeat_interleave(4,dim=1).transpose(0,1)
        scores=(a @ b.transpose(-2,-1))*(64**-.5)
        offsets=torch.arange(n,device=q.device)
        allowed=torch.ones((n,n),device=q.device,dtype=torch.bool)
        if window[0]>=0:allowed &= offsets[None,:]>=offsets[:,None]-window[0]
        if window[1]>=0:allowed &= offsets[None,:]<=offsets[:,None]+window[1]
        if causal:allowed &= offsets[None,:]<=offsets[:,None]
        mask=torch.zeros((16,n,n),device=q.device,dtype=torch.float32).masked_fill(~allowed,-float('inf'))
        if sinks is not None and mode=='key0':mask[:,:,0]+=sinks.float()[:,None]
        # Oracle matches the checkpoint mask then real-key-zero bias operation.
        if mode=='null':
            scores=scores+mask
            scores=torch.cat([scores,sinks.float()[:,None,None].expand(16,n,1)],dim=-1)
            weights=torch.softmax(scores,dim=-1)[...,:n]
            out=weights@c
        else:
            out=F.scaled_dot_product_attention(a,b,c,attn_mask=mask,scale=64**-.5)
        outputs.append(out.transpose(0,1));start+=n
    return torch.cat(outputs)

def case(name,lengths,window,sink,strided=False,causal=False):
    total=sum(lengths)
    if strided:
        fused=torch.randn(total,24,64,device='cuda',dtype=torch.bfloat16)
        q,k,v=fused[:,:16],fused[:,16:20],fused[:,20:]
        assert not q.is_contiguous() and not k.is_contiguous()
    else:
        q=torch.randn(total,16,64,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(total,4,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    starts=torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()),device='cuda',dtype=torch.int32)
    lens=torch.tensor(lengths,device='cuda',dtype=torch.int32)
    sinks=torch.linspace(-3,3,16,device='cuda') if sink else None
    out=torch.empty_like(q)
    context_attention_fwd(q,k,v,out,starts,lens,max(lengths),is_causal=causal,window_size=window,key0_sink=sinks)
    expected=oracle(q,k,v,lengths,window,sinks,causal=causal)
    torch.cuda.synchronize()
    error=(out.float()-expected).abs()
    passed=torch.isfinite(out).all().item() and torch.allclose(out.float(),expected,atol=.035,rtol=.03)
    row={'name':name,'lengths':lengths,'q_heads':16,'kv_heads':4,'dim':64,'window':window,
         'sink':sink,'strided':strided,'causal':causal,'max_abs_error':error.max().item(),
         'rmse':error.square().mean().sqrt().item(),'atol':.035,'rtol':.03,'passed':passed}
    if not sink and window==(-1,-1):
        old=torch.empty_like(q)
        original.context_attention_fwd(q,k,v,old,starts,lens,max(lengths),is_causal=causal)
        row['original_bitwise_equal']=torch.equal(old,out)
        assert row['original_bitwise_equal'],row
    if sink and max(lengths)>128:
        wrongfull=oracle(q,k,v,lengths,(-1,-1),None)
        wrongnull=oracle(q,k,v,lengths,window,sinks,mode='null')
        row['wrong_full_no_sink_max_diff']=(wrongfull-expected).abs().max().item()
        row['wrong_null_sink_max_diff']=(wrongnull-expected).abs().max().item()
        assert row['wrong_full_no_sink_max_diff']>.05 and row['wrong_null_sink_max_diff']>.05,row
    if name=='varlen-strided-window-key0':
        from sglang.srt.layers.attention.vision import VisionTritonAttention
        backend=VisionTritonAttention(use_data_parallel=True)
        metadata=N(cu_seqlens=starts,seq_lens=lens,max_seqlen=max(lengths))
        common=dict(q=q,k=k,v=v,cu_seqlens=starts,bsz=len(lengths),seq_len=max(lengths),
                    forward_metadata=metadata,window_size=window,s_aux=sinks)
        generic=backend.forward(**common)
        row['generic_backend_ignores_new_optin']=torch.allclose(generic.float(),wrongfull,atol=.035,rtol=.03)
        backend.mimo_vision_key0_sink=True
        opted=backend.forward(**common)
        row['mimo_backend_matches_oracle']=torch.allclose(opted.float(),expected,atol=.035,rtol=.03)
        assert row['generic_backend_ignores_new_optin'] and row['mimo_backend_matches_oracle'],row
    rows.append(row)
    if args.output:
        args.output.write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'cases':rows},indent=2)+'\n')
    print(json.dumps(row),flush=True)
    assert passed,row

for n in [1,17,65,129,257,511,1025]:case('window-key0-'+str(n),[n],(64,64),True)
case('varlen-strided-window-key0',[3,137,259],(64,64),True,True)
case('fully-masked-intermediate-tiles',[7,385],(4,4),True,True)
case('window-without-sink',[129,257],(64,64),False,True)
case('full-original-parity',[1,17,129,257],(-1,-1),False,True)
case('causal-original-parity',[17,257],(-1,-1),False,True,True)
print('VISION_KERNEL_PASS '+str(len(rows)),flush=True)
