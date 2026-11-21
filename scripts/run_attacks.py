import sys,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scipy.special import expit
from src.common import *
from src.engine import load_value
from src.paper_models import PaperDetector
from src.attacks import attack
from src.metrics import probability

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modality',choices=['audio','image','video'],default='audio');p.add_argument('--method',choices=['paper','baseline'],default='paper');p.add_argument('--profile',choices=['local','full'],default='local');p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    m=args.modality
    if args.method=='baseline':
        if m!='audio':raise ValueError('Differentiable baseline currently supports audio only')
        from src.audio_baseline_torch import AudioBaselineTorch
        class Wrapper:
            def __init__(self):self.device=torch_setup();self.model=AudioBaselineTorch(self.device)
            def audio_tensor(self,a):return torch.tensor(a,dtype=torch.float32,device=self.device)[None]
            def audio_logit(self,x):return self.model(x)
        model=Wrapper()
    else:model=PaperDetector(m)
    for param in model.model.parameters():param.requires_grad_(False)
    if hasattr(model,'head'):
        for param in model.head.parameters():param.requires_grad_(False)
    rows=[r for r in read('data/manifest.json') if r['modality']==m and r['split']=='test']
    rows=sorted(rows,key=lambda r:(r['id'].split('_')[-1],r['label']))
    if args.limit:
        by_label={label:[r for r in rows if r['label']==label] for label in [0,1]}
        rows=[r for pair in __import__('itertools').zip_longest(by_label[0],by_label[1]) for r in pair if r is not None][:args.limit]
    cal=read(f'models/{m}_calibration.json')[args.method];out=f'results/attacks_{m}_{args.method}_{args.profile}.json'
    results=read(out) if (ROOT/out).exists() else [];done={(r['id'],r['attack'],r['epsilon'],r['defense']) for r in results}
    failure_path=out.replace('.json','_failures.json')
    failures=[]
    epsilons=[.001,.004] if m=='audio' else [2/255,8/255]
    settings=[('FGSM',1,1,False),('PGD5',5,1,True)]
    if args.profile=='full':
        epsilons=[.0005,.001,.002,.004] if m=='audio' else [1/255,2/255,4/255,8/255]
        settings+=[('PGD20_R3',20,3,True)]
    def score(z):return model.audio_logit(z) if m=='audio' else model.image_logit(z)
    def defense(z):
        if m=='audio':return torch.nn.functional.avg_pool1d(torch.nn.functional.pad(z[:,None],(1,1),mode='reflect'),3,stride=1)[:,0]
        return torch.nn.functional.avg_pool2d(torch.nn.functional.pad(z,(1,1,1,1),mode='reflect'),3,stride=1)
    for row in rows:
        try:value=load_value(m,ROOT/row['path'])
        except Exception as error:
            failures.append(dict(id=row['id'],label=row['label'],group=row['group'],error=str(error),stage='decode_or_face_detection'))
            dump(failure_path,failures)
            print('ATTACK INPUT FAILED',m,row['id'],str(error),flush=True)
            continue
        if m=='audio':x=model.audio_tensor(value);lower,upper=-1.,1.
        else:x=model.image_tensor(value[0]['face'] if m=='video' else value);lower,upper=0.,1.
        for defended in [False,True]:
            fn=(lambda z:score(defense(z))) if defended else score
            with torch.no_grad():clean_raw=float(fn(x).item());clean_p=float(probability(clean_raw,cal))
            for epsilon in epsilons:
                for name,steps,restarts,random_start in settings:
                    key=(row['id'],name,epsilon,defended)
                    if key in done:continue
                    t=time.perf_counter();adv=attack(fn,x,row['label'],epsilon,steps,restarts,random_start,lower,upper)
                    with torch.no_grad():adv_raw=float(fn(adv).item())
                    adv_p=float(probability(adv_raw,cal));delta=adv-x
                    r=dict(id=row['id'],group=row['group'],label=row['label'],attack=name,epsilon=epsilon,steps=steps,restarts=restarts,
                           defense=defended,adaptive_attack=defended,clean_probability=clean_p,adv_probability=adv_p,
                           clean_correct=int(clean_p>=.5)==row['label'],adv_correct=int(adv_p>=.5)==row['label'],
                           linf=float(delta.abs().max()),l2=float(delta.norm()),snr_db=float(10*torch.log10(x.square().mean()/(delta.square().mean()+1e-20))),seconds=time.perf_counter()-t,
                           scope='full waveform LFCC baseline' if args.method=='baseline' else 'first model segment only' if m=='audio' else 'first detected face crop only' if m=='video' else 'model-input crop; no file re-encoding')
                    results.append(r);dump(out,results)
                    print('ATTACK',m,row['id'],name,epsilon,defended,round(r['seconds'],2),flush=True)
                    evidence=ROOT/f'results/attack_examples/{m}_{args.method}_{row["id"]}_{name}_{epsilon}_{defended}.npz';evidence.parent.mkdir(exist_ok=True)
                    np.savez_compressed(evidence,clean=x.detach().cpu().numpy(),adversarial=adv.cpu().numpy())
    summary=[]
    for defense_flag in [False,True]:
        for epsilon in epsilons:
            for name,*_ in settings:
                selected=[r for r in results if r['attack']==name and r['epsilon']==epsilon and r['defense']==defense_flag]
                eligible=[r for r in selected if r['clean_correct']]
                summary.append(dict(attack=name,epsilon=epsilon,defense=defense_flag,n=len(selected),eligible=len(eligible),
                                    asr=sum(not r['adv_correct'] for r in eligible)/len(eligible) if eligible else None,
                                    accuracy=sum(r['adv_correct'] for r in selected)/len(selected) if selected else None,
                                    false_positive_attacks=dict(eligible=sum(r['label']==0 for r in eligible),success=sum(r['label']==0 and not r['adv_correct'] for r in eligible)),
                                    false_negative_attacks=dict(eligible=sum(r['label']==1 for r in eligible),success=sum(r['label']==1 and not r['adv_correct'] for r in eligible))))
    dump(failure_path,failures)
    dump(f'results/attacks_{m}_{args.method}_{args.profile}_summary.json',dict(modality=m,method=args.method,profile=args.profile,results=summary,requested_samples=len(rows),input_failures=failures,calibration='clean validation calibration retained for defense; no defense-specific recalibration',scope=f'model input threat model, {args.profile} profile; not a claim of certified robustness'))
