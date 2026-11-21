import sys,json,importlib.metadata,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *

if __name__=='__main__':
    sources=read('models/sources.json');existing={r['path'] for r in sources}
    additions=[('models/distilgpt2/model.safetensors','https://huggingface.co/distilbert/distilgpt2/resolve/main/model.safetensors'),('models/ViT-L-14.pt','https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt')]
    for path,url in additions:
        if path not in existing:
            p=ROOT/path;sources.append(dict(path=path,url=url,bytes=p.stat().st_size,sha256=digest(p)))
    assert next(r for r in sources if r['path']=='models/ViT-L-14.pt')['sha256']=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    dump('models/sources.json',sources)
    course=read('data/manifest.json')
    for path in (ROOT/'results').glob('*_metrics.json'):
        obj=read(path);n=sum(r['modality']==obj['modality'] and r['split']=='test' for r in course)
        obj['requested_test_count']=n;obj['test_coverage']=obj['test']['n']/n
        obj['effective_accuracy_counting_failures_as_wrong']=round(obj['test']['accuracy']*obj['test']['n'])/n
        dump(path,obj)
    names=['numpy','scipy','scikit-learn','pandas','Pillow','opencv-python','matplotlib','soundfile','gradio','transformers','huggingface-hub','safetensors','ftfy','regex','tqdm','pyarrow']
    locked=[]
    for name in names:
        try:locked.append(name+'=='+importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:pass
    (ROOT/'requirements-lock-core.txt').write_text('\n'.join(locked)+'\n',encoding='utf-8')
    paths=[ROOT/'app.py',ROOT/'requirements.txt']+list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    dump('results/code_manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in paths})
    all_rows=read('data/manifest.json')+read('data/external_manifest.json')+read('data/text_selection_manifest.json')
    if (ROOT/'data/external_visual_manifest.json').exists():
        all_rows+=read('data/external_visual_manifest.json')
    hashes=[r['sha256'] for r in all_rows];assert len(hashes)==len(set(hashes))
    for row in all_rows:assert digest(ROOT/row['path'])==row['sha256']
    ui=read('results/ui_verification.json') if (ROOT/'results/ui_verification.json').exists() else {}
    selected=read('configs/selected_methods.json')
    matches=all(any(r['modality']==m and r['method']==obj['selected'] for r in ui.get('detections',[])) for m,obj in selected.items())
    status=dict(total_acquired_unique_samples=len(all_rows),course=348,external=len(all_rows)-348,ui_default_methods_verified=matches,
                four_modalities=True,four_layer_output=True,baseline_and_paper=True,validation_method_selection=read('configs/selected_methods.json'),
                robustness_files=[p.name for p in (ROOT/'results').glob('robustness_*_summary.json')],
                attack_files=[p.name for p in (ROOT/'results').glob('attacks_*_summary.json')],
                limitations=['Visual UI inspection unavailable because computer-use kernel failed; API verification is separate.',
                'Image/video attacks are model-input/crop-level; inspect each profile summary for actual sample count and coverage.',
                'CIFAKE external images are 32x32 and cover one generator; no external video corpus acquired.',
                'No adversarial training performed; smoothing defense and adaptive attacks evaluated.'])
    status['ordinary_robustness_records']=sum(len(read(p)) for p in (ROOT/'results').glob('robustness_*.json') if not p.name.endswith('_summary.json'))
    status['adversarial_records']=sum(len(read(p)) for p in (ROOT/'results').glob('attacks_*.json') if not p.name.endswith(('_summary.json','_failures.json')))
    status['counting_note']='Perturbed/attacked records are repeated evaluations of original samples, not additional independent samples.'
    dump('results/completion_audit.json',status);print('Audit complete; unique samples',len(all_rows),flush=True)
