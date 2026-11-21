"""Fetch official research code/weights. No remote Python code is executed here."""
from pathlib import Path
import urllib.request, json, hashlib, re, time
ROOT=Path(__file__).resolve().parents[1]
records=[]
def fetch(url,path):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url,timeout=180) as src, p.with_suffix(p.suffix+'.part').open('wb') as out:
                    while chunk:=src.read(1024*1024):out.write(chunk)
                p.with_suffix(p.suffix+'.part').replace(p);break
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    records.append(dict(path=path,url=url,bytes=p.stat().st_size,sha256=h.hexdigest()))
    (ROOT/'models/sources.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(path,p.stat().st_size,flush=True)
    return p

if __name__=='__main__':
    for f in ['models/AASIST.py','config/AASIST-L.conf','models/weights/AASIST-L.pth','LICENSE','README.md']:
        fetch('https://raw.githubusercontent.com/clovaai/aasist/main/'+f,'vendor/aasist/'+f)
    for f in ['model.py']:
        fetch('https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/models/clip/'+f,'vendor/univfd/'+f)
    fetch('https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth','models/univfd_fc.pth')
    fetch('https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/LICENSE','vendor/univfd/LICENSE')
    p=fetch('https://raw.githubusercontent.com/SCLBD/DeepfakeBench/main/training/networks/xception.py','vendor/deepfakebench/xception_original.py')
    source=p.read_text('utf-8').replace('from metrics.registry import BACKBONE','').replace('@BACKBONE.register_module(module_name="xception")','')
    (p.parent/'xception.py').write_text('# Adaptation: removed framework-only registry; architecture unchanged.\n'+source,encoding='utf-8')
    fetch('https://raw.githubusercontent.com/SCLBD/DeepfakeBench/main/LICENSE','vendor/deepfakebench/LICENSE')
    fetch('https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth','models/xception_best.pth')
    for f in ['config.json','tokenizer_config.json','vocab.json','merges.txt','tokenizer.json','model.safetensors']:
        fetch('https://huggingface.co/distilbert/distilgpt2/resolve/main/'+f,'models/distilgpt2/'+f)
    fetch('https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt','models/ViT-L-14.pt')
    assert records[-1]['sha256']=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
