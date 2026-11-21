"""Verify copied assets and sample inference/input gradients; no claim of GPU execution until run there."""
import sys,argparse,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.common import *
from src.paper_models import PaperDetector
from src.engine import load_value
from src.attacks import attack

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-cuda',action='store_true');parser.add_argument('--skip-hashes',action='store_true');args=parser.parse_args()
    if args.require_cuda and (device()!='cuda' or not torch.cuda.is_available()):raise SystemExit('CUDA required. Set LAB_DEVICE=cuda and install a compatible CUDA PyTorch.')
    report=dict(python=platform.python_version(),torch=torch.__version__,device=device(),cuda_available=torch.cuda.is_available(),checks=[])
    if torch.cuda.is_available():report.update(gpu=torch.cuda.get_device_name(0),torch_cuda=torch.version.cuda)
    if not args.skip_hashes:
        for filename in ['data/manifest.json','data/external_manifest.json','data/text_selection_manifest.json']:
            for r in read(filename):
                if digest(ROOT/r['path'])!=r['sha256']:raise RuntimeError('Data hash mismatch: '+r['path'])
        for r in read('models/sources.json'):
            if digest(ROOT/r['path'])!=r['sha256']:raise RuntimeError('Model/vendor source hash mismatch: '+r['path'])
        report['asset_hashes_verified']=True
    rows=read('data/manifest.json')
    for m in MODALITIES:
        row=next(r for r in rows if r['modality']==m and r['split']=='test' and r['label']==0)
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter();model=PaperDetector(m);value=load_value(m,ROOT/row['path']);raw,details=model.score(value)
        item=dict(modality=m,id=row['id'],raw_score=raw,model_device=str(next(model.model.parameters()).device))
        if (ROOT/'GPU_WORKSPACE.json').exists():
            parent=Path(read('GPU_WORKSPACE.json')['parent_snapshot'])
            reference=parent/f'results/{m}_paper_scores.json'
            if reference.exists():
                records=json.loads(reference.read_text('utf-8'));prior=next((r for r in records if r['id']==row['id'] and 'raw_score' in r),None)
                if prior:item.update(cpu_reference_raw=prior['raw_score'],absolute_cpu_gpu_difference=abs(raw-prior['raw_score']))
        if m in ['image','audio','video']:
            x=model.audio_tensor(value) if m=='audio' else model.image_tensor(value[0]['face'] if m=='video' else value)
            fn=model.audio_logit if m=='audio' else model.image_logit
            adv=attack(fn,x,row['label'],.0005 if m=='audio' else 1/255,steps=1,random_start=False,lower=-1 if m=='audio' else 0)
            item['input_gradient_verified']=bool(torch.isfinite(adv).all());item['actual_linf']=float((adv-x).abs().max())
        item['seconds']=time.perf_counter()-t
        if torch.cuda.is_available():item['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        report['checks'].append(item);model.close();del model
        print('PREFLIGHT',m,item,flush=True)
    dump('results/gpu_preflight.json',report)
