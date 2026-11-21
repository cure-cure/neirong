import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score,average_precision_score,brier_score_loss

def calibrate(scores,labels):
    x=np.asarray(scores);y=np.asarray(labels)
    def loss(v):
        z=v[0]*x+v[1];return float(np.mean(np.logaddexp(0,z)-y*z)+.001*(v[0]-1)**2)
    fit=minimize(loss,[1.,0.],bounds=[(.0001,20),(-30,30)],method='L-BFGS-B')
    return dict(slope=float(fit.x[0]),intercept=float(fit.x[1]),n=len(y),method='positive-slope logistic calibration on validation only',success=bool(fit.success))
def probability(scores,cal):return expit(cal['slope']*np.asarray(scores)+cal['intercept'])
def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pred=p>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();pr,re,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    confidence=np.maximum(p,1-p);correct=pred==y;ece=0
    for lo in np.linspace(.5,.95,10):
        mask=(confidence>=lo)&(confidence<lo+.05+1e-10)
        if mask.any():ece+=mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return dict(n=int(len(y)),accuracy=float(accuracy_score(y,pred)),precision=float(pr),recall=float(re),f1=float(f),
                fpr=float(fp/max(1,tn+fp)),fnr=float(fn/max(1,tp+fn)),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],
                roc_auc=float(roc_auc_score(y,p)) if len(set(y))>1 else None,
                pr_auc=float(average_precision_score(y,p)) if len(set(y))>1 else None,brier=float(brier_score_loss(y,p)),ece=float(ece))

def bootstrap_ci(y,p,groups,repeats=500):
    rng=np.random.default_rng(202430);groups=np.array(groups);unique=np.unique(groups);y=np.array(y);p=np.array(p);values=[]
    for _ in range(repeats):
        chosen=rng.choice(unique,len(unique),replace=True);idx=np.concatenate([np.flatnonzero(groups==g) for g in chosen]);values.append(float(np.mean((p[idx]>=.5)==y[idx])))
    return [float(v) for v in np.quantile(values,[.025,.975])]
