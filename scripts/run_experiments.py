import sys,argparse,time,joblib,traceback,platform,importlib.metadata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.engine import load_value,features
from src.metrics import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def evaluate(modality,method):
    rows=[r for r in read('data/manifest.json') if r['modality']==modality];out=ROOT/f'results/{modality}_{method}_scores.json'
    scores=read(out) if out.exists() else [];done={r['id'] for r in scores if 'error' not in r}
    scores=[r for r in scores if r['id'] in done]
    if method=='baseline':
        cache=ROOT/f'data/processed/{modality}_features.joblib'
        if cache.exists():values=joblib.load(cache)
        else:
            values={}
            for i,row in enumerate(rows):
                try:values[row['id']]=features(modality,load_value(modality,ROOT/row['path']))
                except Exception as e:values[row['id']]={'error':str(e)}
                if (i+1)%10==0:print(modality,'features',i+1,flush=True)
            joblib.dump(values,cache)
        train=[r for r in rows if r['split']=='train' and not isinstance(values[r['id']],dict)]
        x=[values[r['id']] for r in train];y=[r['label'] for r in train]
        if modality=='text':model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=20000,sublinear_tf=True),LogisticRegression(C=1,max_iter=1000,random_state=SEED))
        else:model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=SEED))
        model.fit(x,y);joblib.dump(model,ROOT/f'models/{modality}_baseline.joblib')
    else:
        from src.paper_models import PaperDetector
        model=PaperDetector(modality)
    for i,row in enumerate(rows):
        if row['id'] in done:continue
        record=dict(row);t=time.perf_counter()
        try:
            if method=='baseline':
                value=values[row['id']]
                if isinstance(value,dict):raise ValueError(value['error'])
                record['raw_score']=float(model.decision_function([value])[0]);record['details']={}
            else:record['raw_score'],record['details']=model.score(load_value(modality,ROOT/row['path']))
        except Exception as e:record['error']=str(e);traceback.print_exc()
        record['seconds']=time.perf_counter()-t;scores.append(record);dump(out,scores)
        print(modality,method,i+1,'/',len(rows),record.get('raw_score',record.get('error')),round(record['seconds'],2),flush=True)
    if method=='paper':model.close()
    valid=[r for r in scores if 'error' not in r];val=[r for r in valid if r['split']=='val'];test=[r for r in valid if r['split']=='test']
    cal=calibrate([r['raw_score'] for r in val],[r['label'] for r in val]);cp=ROOT/f'models/{modality}_calibration.json'
    c=read(cp) if cp.exists() else {};c[method]=cal;dump(cp,c)
    for r in scores:
        if 'error' not in r:r['fake_probability']=float(probability(r['raw_score'],cal))
    dump(out,scores)
    y=[r['label'] for r in test];p=probability([r['raw_score'] for r in test],cal)
    summary=dict(modality=modality,method=method,test=metrics(y,p),raw_test=metrics(y,expit([r['raw_score'] for r in test])),
                 calibration=cal,coverage=len(valid)/len(rows),errors=[r for r in scores if 'error' in r],
                 description='Adapted Fast-DetectGPT with DistilGPT2' if modality=='text' and method=='paper' else method,
                 test_ids=[r['id'] for r in test],accuracy_ci95=bootstrap_ci(y,p,[r['group'] for r in test]),
                 inference_seconds=sum(r['seconds'] for r in scores),scope='course subset grouped held-out test; NOT independent external benchmark')
    requested=sum(r['split']=='test' for r in rows)
    summary['requested_test_count']=requested
    summary['test_coverage']=len(test)/requested
    summary['effective_accuracy_counting_failures_as_wrong']=float(np.sum((p>=.5)==np.asarray(y))/requested)
    dump(f'results/{modality}_{method}_metrics.json',summary)
    print('SUMMARY',modality,method,summary['test'],flush=True)

def combined(modality):
    a=read(f'results/{modality}_baseline_scores.json');b={r['id']:r for r in read(f'results/{modality}_paper_scores.json') if 'error' not in r}
    rows=[]
    for r in a:
        if r['id'] in b and 'error' not in r:
            t=dict(r);t['fake_probability']=(r['fake_probability']+b[r['id']]['fake_probability'])/2;rows.append(t)
    dump(f'results/{modality}_combined_scores.json',rows);test=[r for r in rows if r['split']=='test']
    result=metrics([r['label'] for r in test],[r['fake_probability'] for r in test])
    dump(f'results/{modality}_combined_metrics.json',dict(modality=modality,method='combined',test=result,coverage=len(rows)/len(a),accuracy_ci95=bootstrap_ci([r['label'] for r in test],[r['fake_probability'] for r in test],[r['group'] for r in test]),scope='equal mean of validation-calibrated baseline and paper scores; no test-set model selection'))
    print('COMBINED',modality,result,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--modality',choices=list(MODALITIES)+['all'],default='all');parser.add_argument('--method',choices=['baseline','paper','combined','all'],default='baseline');args=parser.parse_args();seed()
    packages={}
    for p in ['torch','torchvision','numpy','scikit-learn','gradio','transformers','soundfile']:
        try:packages[p]=importlib.metadata.version(p)
        except Exception:pass
    dump('results/environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages=packages,device=os.environ.get('LAB_DEVICE','cpu'),seed=SEED,threads=os.environ.get('LAB_THREADS','4')))
    for m in MODALITIES if args.modality=='all' else [args.modality]:
        for method in ['baseline','paper','combined'] if args.method=='all' else [args.method]:
            if method=='combined':combined(m)
            else:evaluate(m,method)
