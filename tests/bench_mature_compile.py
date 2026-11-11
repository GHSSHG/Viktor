"""Two-rank, identical-data/weights eager vs compile benchmark, including backward,
accumulation, allreduce, global clip and Adam. No production checkpoints changed.
Real labels/signals; synthetic monotone alignment centres exercise the same KL
shapes in both modes (not an accuracy experiment). Timings exclude compilation.
"""
import argparse,contextlib,json,os,time,gc
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.train import fixed_windows,named_groups
from scripts.train_mature import build_model,Trainable,convert_centers
from viktor.data import collate


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--compression',type=int,required=True)
 p.add_argument('--model-dir',required=True)
 p.add_argument('--encoder-init',required=True)
 p.add_argument('--windows',required=True)
 p.add_argument('--out',required=True)
 p.add_argument('--phase',choices=['frozen','joint'],required=True)
 a=p.parse_args()
 a.seed,a.tokenizer,a.layers=0,'4mer',4
 a.prefix_pos_scale,a.align_sigma=a.compression,4/a.compression
 dist.init_process_group('nccl')
 rank=dist.get_rank()
 dev=torch.device('cuda',int(os.environ['LOCAL_RANK']))
 torch.cuda.set_device(dev)
 torch.set_num_threads(2)
 torch._inductor.config.split_reductions=False
 # Nested compiler under DDP must not split the loss graph by DDP buckets.
 torch._dynamo.config.optimize_ddp=False
 torch.backends.cuda.matmul.allow_tf32=True
 windows=fixed_windows(a.windows,'train',64,23.506745239082388,0.,'part2') if False else None
 # Use the Encoder's actual standardisation for real training windows.
 seed_model=build_model(a,dev)
 windows=fixed_windows(a.windows,'train',64,seed_model.encoder.mean,seed_model.encoder.stdev,'part2')
 del seed_model
 micros=[]
 model=build_model(a,dev)
 for start in (rank*32,rank*32+16):
  sig,valid,seqs,ws=collate(windows[start:start+16])
  cs=[np.linspace(0,(int(v)+11)//12-1,len(s),dtype=np.float32) for v,s in zip(valid,seqs)]
  data=model.batch(seqs,ws,128,convert_centers(cs,a.compression))
  micros.append(tuple(t.to(dev) for t in (sig,valid,*data)))
 del model
 denom=torch.tensor([sum(float(m[4].sum()) for m in micros),sum(int((m[5]>=0).sum()) for m in micros)],device=dev)
 dist.all_reduce(denom)
 results={}
 ref=None
 for compiled in (False,True):
  model=build_model(a,dev)
  model.encoder.set_base_trainable(a.phase=='joint')
  ddp=DDP(Trainable(model,compiled),device_ids=[dev.index],broadcast_buffers=False)
  opt=torch.optim.AdamW(named_groups(model,'all',.1),lr=0.,betas=(.9,.95),fused=True)
  def step():
   opt.zero_grad(set_to_none=True)
   values=[]
   for i,m in enumerate(micros):
    with ddp.no_sync() if i==0 else contextlib.nullcontext():
     with torch.autocast('cuda',dtype=torch.bfloat16):
      _,ce,align,_,_=ddp(*m)
      loss=ce*(2*m[4].sum()/denom[0])+.1*align*(2*(m[5]>=0).sum()/denom[1])
     loss.backward()
     values.extend((ce.detach(),align.detach()))
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   opt.step()
   return torch.stack(values)
  vals=step()
  if compiled:
   torch.testing.assert_close(vals,ref[0],atol=.015,rtol=.015)
   differences=[]
   for n,p in model.named_parameters():
    if p.grad is not None:
     expected=ref[1][n].to(dev)
     relative=float((p.grad.float()-expected).norm()/expected.norm().clamp_min(1e-12))
     differences.append(relative)
     assert relative<.08,(n,relative)
   if rank==0:print('PARITY PASS',a.phase,'max_param_grad_relative',max(differences),flush=True)
  else:
   ref=(vals.clone(),{n:p.grad.detach().float().cpu() for n,p in model.named_parameters() if p.grad is not None})
  for _ in range(5):step()
  timings=[]
  for _ in range(3):
   dist.barrier();torch.cuda.synchronize()
   begin=time.perf_counter()
   for i in range(15):step()
   torch.cuda.synchronize()
   elapsed=torch.tensor((time.perf_counter()-begin)/15,device=dev)
   dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
   timings.append(float(elapsed))
  results['compiled' if compiled else 'eager']=timings
  if rank==0:print('BENCH',a.compression,a.phase,compiled,timings,flush=True)
  del ddp,opt,model
  gc.collect();torch.cuda.empty_cache()
 results['time_reduction']=1-float(np.median(results['compiled']))/float(np.median(results['eager']))
 if rank==0:
  Path(a.out).write_text(json.dumps(results,indent=2))
  print('RESULT',json.dumps(results),flush=True)
 dist.destroy_process_group()

if __name__=='__main__':main()
