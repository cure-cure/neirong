"""Choose deployed methods from validation data only; never rank by test metrics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.metrics import metrics

if __name__=='__main__':
    selection={}
    for modality in MODALITIES:
        candidates={}
        for method in ['baseline','paper','combined']:
            p=ROOT/f'results/{modality}_{method}_scores.json'
            if not p.exists():continue
            rows=[r for r in read(p) if r['split']=='val' and 'error' not in r and 'fake_probability' in r]
            if not rows:continue
            value=metrics([r['label'] for r in rows],[r['fake_probability'] for r in rows])
            value['balanced_accuracy']=1-(value['fpr']+value['fnr'])/2;candidates[method]=value
            external=ROOT/f'results/external_validation_{modality}_{method}_summary.json'
            if external.exists():
                e=read(external)['metrics'];value['external_validation_balanced_accuracy']=1-(e['fpr']+e['fnr'])/2
                value['selection_score']=(value['balanced_accuracy']+value['external_validation_balanced_accuracy'])/2
        if not candidates:continue
        # Prefer simpler method when performance ties. Combined is an ablation, not the default selection pool.
        eligible=[m for m in ['baseline','paper'] if m in candidates]
        use_external=all('selection_score' in candidates[m] for m in eligible)
        criterion='selection_score' if use_external else 'balanced_accuracy'
        chosen=max(eligible,key=lambda m:(round(candidates[m][criterion],6),m=='baseline'))
        selection[modality]=dict(selected=chosen,validation=candidates,rule='equal domain-weighted course/external validation balanced accuracy; ties baseline' if use_external else 'highest course validation balanced accuracy; ties prefer baseline; no test metrics used',
                                 paper_gap_to_baseline=(candidates['paper']['balanced_accuracy']-candidates['baseline']['balanced_accuracy']) if 'paper' in candidates and 'baseline' in candidates else None,
                                 status='final for current available candidates' if 'paper' in candidates else 'provisional; paper evaluation pending')
        selection[modality]['reason']=f'{chosen} has selection score {candidates[chosen][criterion]:.4f}; criterion={criterion}. Paper methods are retained as comparisons even when baseline is deployed.'
    dump('configs/selected_methods.json',selection)
    for m,v in selection.items():print(m,v['selected'],v['paper_gap_to_baseline'],v['status'])
