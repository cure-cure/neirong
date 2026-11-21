import sys,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.engine import Engine,load_value
from src.metrics import metrics
from src.perturbations import CONDITIONS,transform

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modality',default='all',choices=list(MODALITIES)+['all']);p.add_argument('--method',default='baseline',choices=['baseline','paper','combined']);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    engine=Engine()
    for m in MODALITIES if args.modality=='all' else [args.modality]:
        rows=[r for r in read('data/manifest.json') if r['modality']==m and r['split']=='test']
        if args.limit:rows=sorted(rows,key=lambda r:(r['id'].split('_')[-1],r['label']))[:args.limit]
        out=f'results/robustness_{m}_{args.method}.json';path=ROOT/out;results=read(out) if path.exists() else [];done={(r['id'],r['condition'],r['level']) for r in results if 'error' not in r}
        for row in rows:
            try:value=load_value(m,ROOT/row['path'])
            except Exception as e:results.append(dict(id=row['id'],condition='decode',level=0,error=str(e)));continue
            for name,level in [('clean',0)]+CONDITIONS[m]:
                if (row['id'],name,level) in done:continue
                r=dict(id=row['id'],group=row['group'],label=row['label'],condition=name,level=level)
                try:
                    changed=value if name=='clean' else transform(m,value,name,level)
                    result=engine.infer(m,changed,args.method);r.update(fake_probability=result['fake_probability'],seconds=result['seconds'])
                except Exception as e:r['error']=str(e);traceback.print_exc()
                results.append(r);dump(out,results)
            print('ROBUST',m,args.method,row['id'],flush=True)
        summary=[]
        for name,level in [('clean',0)]+CONDITIONS[m]:
            subset=[r for r in results if r['condition']==name and r['level']==level and 'error' not in r]
            if subset:summary.append(dict(condition=name,level=level,**metrics([r['label'] for r in subset],[r['fake_probability'] for r in subset])))
        dump(f'results/robustness_{m}_{args.method}_summary.json',dict(modality=m,method=args.method,results=summary,scope='held-out course subset; video perturbations apply to extracted face crops, NOT end-to-end video re-encoding',requested=len(rows),errors=[r for r in results if 'error' in r]))
        if engine.paper:engine.paper.close();engine.paper=None;engine.current=None

