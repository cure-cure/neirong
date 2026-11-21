"""Nuisance-only controls: can superficial dataset properties predict labels?"""
import sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,soundfile as sf,cv2
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.common import *
from src.metrics import metrics

def nuisance(row):
    p=ROOT/row['path'];m=row['modality']
    if m=='text':
        s=p.read_text('utf-8-sig');words=re.findall(r'\w+',s);return [len(s),len(words),len(set(w.lower() for w in words))/max(1,len(words)),s.count('\n')]
    if m=='image':
        with Image.open(p) as im:return [im.width,im.height,int(im.format=='PNG'),int(im.format=='JPEG')]
    if m=='audio':
        a,sr=sf.read(p,always_2d=True);return [len(a)/sr,sr,a.shape[1],float(np.sqrt(np.mean(a*a)))]
    cap=cv2.VideoCapture(str(p))
    try:return [cap.get(3),cap.get(4),cap.get(5),cap.get(7)/max(1,cap.get(5))]
    finally:cap.release()

if __name__=='__main__':
    rows=read('data/manifest.json');result={}
    for m in MODALITIES:
        subset=[r for r in rows if r['modality']==m];values={r['id']:nuisance(r) for r in subset}
        train=[r for r in subset if r['split']=='train'];test=[r for r in subset if r['split']=='test']
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=SEED))
        model.fit([values[r['id']] for r in train],[r['label'] for r in train]);probs=model.predict_proba([values[r['id']] for r in test])[:,1]
        result[m]=dict(metrics=metrics([r['label'] for r in test],probs),
                       features={'text':'length, word count, vocabulary diversity, newlines','image':'width, height, PNG/JPEG format','audio':'duration, sample rate, channels, RMS','video':'width, height, FPS, duration'}[m],
                       class_means={str(label):np.mean([values[r['id']] for r in subset if r['label']==label],axis=0).tolist() for label in [0,1]},
                       note='Nuisance-only audit model is NOT deployed; high accuracy indicates potential dataset shortcuts, not direct proof that the main model uses these properties.')
        print('Shortcut control',m,result[m]['metrics']['accuracy'],flush=True)
    dump('results/shortcut_audit.json',result)
