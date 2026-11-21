import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    directory=ROOT/'results/figures';directory.mkdir(exist_ok=True)
    for path in sorted((ROOT/'results').glob('robustness_*_summary.json')):
        obj=read(path);rows=obj['results'];fig,ax=plt.subplots(figsize=(11,4))
        x=np.arange(len(rows));ax.bar(x,[r['accuracy'] for r in rows],color='#3276ae')
        ax.set(xticks=x,xticklabels=[f'{r["condition"]}\n{r["level"]}' for r in rows],ylim=(0,1.1),ylabel='Accuracy',title=f'{obj["modality"]} / {obj["method"]} / held-out perturbation tests')
        for i,r in enumerate(rows):ax.text(i,r['accuracy']+.025,f'{r["accuracy"]:.2f}\nn={r["n"]}',ha='center',fontsize=7)
        ax.tick_params(axis='x',labelsize=8);fig.tight_layout();fig.savefig(directory/(path.stem.replace('_summary','')+'.png'),dpi=140);plt.close(fig)
    for path in sorted((ROOT/'results').glob('attacks_*_summary.json')):
        obj=read(path);fig,ax=plt.subplots(figsize=(8,4));rows=obj['results']
        for name in sorted(set(r['attack'] for r in rows)):
            for defense in [False,True]:
                values=sorted([r for r in rows if r['attack']==name and r['defense']==defense and r['asr'] is not None],key=lambda r:r['epsilon'])
                if values:ax.plot([r['epsilon'] for r in values],[r['asr'] for r in values],marker='o',linestyle='--' if defense else '-',label=name+(' + adaptive smooth defense' if defense else ''))
        ax.set(xlabel='L-infinity epsilon',ylabel='Conditional attack success rate',ylim=(-.05,1.08),title=path.stem)
        if ax.lines:ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(directory/(path.stem.replace('_summary','')+'.png'),dpi=140);plt.close(fig)
    patterns=['audio_baseline_*PGD20_R3*False.npz','image_paper_*PGD5*False.npz','video_paper_*PGD5*False.npz']
    for pattern in patterns:
        files=sorted((ROOT/'results/attack_examples').glob(pattern))
        if not files:continue
        path=files[0];obj=np.load(path);a=obj['clean'][0];b=obj['adversarial'][0];d=b-a;fig,axes=plt.subplots(1,3,figsize=(12,3))
        if a.ndim==3:
            axes[0].imshow(a.transpose(1,2,0));axes[1].imshow(b.transpose(1,2,0));axes[2].imshow(np.abs(d).mean(0),cmap='magma')
        else:
            for ax,y in zip(axes,[a,b,d]):ax.plot(np.arange(len(y))/16000,y,linewidth=.4)
        for ax,title in zip(axes,['Clean model input','Adversarial model input','Perturbation; independent display scale']):ax.set_title(title,fontsize=9)
        fig.suptitle(path.stem,fontsize=8);fig.tight_layout();fig.savefig(directory/('example_'+path.stem.split('_')[0]+'.png'),dpi=140);plt.close(fig)
    selection=read('configs/selected_methods.json');reliability=[]
    fig,axes=plt.subplots(2,2,figsize=(10,9));edges=np.linspace(.5,1,6)
    for ax,(modality,details) in zip(axes.flat,selection.items()):
        method=details['selected'];path=ROOT/f'results/{modality}_{method}_scores.json'
        rows=[r for r in read(path) if r.get('split')=='test' and 'error' not in r]
        probabilities=np.asarray([r['fake_probability'] for r in rows]);labels=np.asarray([r['label'] for r in rows])
        confidence=np.maximum(probabilities,1-probabilities);correct=(probabilities>=.5)==labels
        points=[]
        for index,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=(confidence>=lo)&(confidence<=hi if index==len(edges)-2 else confidence<hi)
            if mask.any():
                point=dict(modality=modality,method=method,lower=float(lo),upper=float(hi),n=int(mask.sum()),mean_confidence=float(confidence[mask].mean()),accuracy=float(correct[mask].mean()))
                points.append(point);reliability.append(point)
        ax.plot([.5,1],[.5,1],color='#777',linestyle=':',label='ideal')
        ax.plot([p['mean_confidence'] for p in points],[p['accuracy'] for p in points],marker='o',color='#2369a2',label='observed')
        for p in points:ax.annotate(f"n={p['n']}",(p['mean_confidence'],p['accuracy']),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set(xlim=(.48,1.02),ylim=(-.03,1.05),xlabel='Mean predicted confidence',ylabel='Empirical accuracy',title=f'{modality} / {method} (held-out n={len(rows)})')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Reliability diagrams for validation-selected methods\nFive bins; small held-out samples make estimates uncertain')
    fig.tight_layout();fig.savefig(directory/'reliability_selected_methods.png',dpi=150);plt.close(fig)
    external=ROOT/'results/cifake_external_image_selected_scores.json'
    if external.exists():
        rows=[r for r in read(external) if 'error' not in r];probabilities=np.asarray([r['fake_probability'] for r in rows]);labels=np.asarray([r['label'] for r in rows]);confidence=np.maximum(probabilities,1-probabilities);correct=(probabilities>=.5)==labels
        fig,ax=plt.subplots(figsize=(5.5,5));points=[]
        for index,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=(confidence>=lo)&(confidence<=hi if index==len(edges)-2 else confidence<hi)
            if mask.any():points.append((float(confidence[mask].mean()),float(correct[mask].mean()),int(mask.sum())))
        ax.plot([.5,1],[.5,1],color='#777',linestyle=':');ax.plot([p[0] for p in points],[p[1] for p in points],marker='o')
        for x,y,n in points:ax.annotate(f'n={n}',(x,y),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set(xlim=(.48,1.02),ylim=(-.03,1.05),xlabel='Mean predicted confidence',ylabel='Empirical accuracy',title=f'CIFAKE external / selected (n={len(rows)})');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(directory/'reliability_cifake_external.png',dpi=150);plt.close(fig)
    dump('results/reliability_bins.json',reliability)
    print('Evidence figures updated',flush=True)

if __name__=='__main__':main()
