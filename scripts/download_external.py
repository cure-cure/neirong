"""Acquire supplemental public test data; never add it to training/calibration."""
import sys,urllib.request,json,hashlib,concurrent.futures,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read,dump,digest

if __name__=='__main__':
    rows=[];known={r['sha256'] for r in read('data/manifest.json')};skipped=[]
    for kind,label in [('webtext',0),('small-117M-k40',1)]:
        url=f'https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/{kind}.test.jsonl'
        with urllib.request.urlopen(url,timeout=60) as response:
            for index,line in enumerate(response):
                obj=json.loads(line);text=obj['text'];data=text.encode('utf-8');h=hashlib.sha256(data).hexdigest()
                if h in known:continue
                path=ROOT/f'data/raw/external/text/{kind}_{index:04d}.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);known.add(h)
                rows.append(dict(id=f'external_text_{kind}_{index}',modality='text',label=label,path=path.relative_to(ROOT).as_posix(),group=f'{kind}_{index}',source=url,sha256=h,split='external_test'))
                if sum(r['modality']=='text' and r['label']==label for r in rows)>=100:break
        print('External text',kind,'collected',flush=True)
    base='https://huggingface.co/datasets/rijal028/secureaudio-corpus-standardized/resolve/main/'
    files=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/rijal028/secureaudio-corpus-standardized/tree/main?recursive=true&limit=1000',timeout=45))
    files=[f for f in files if f['path'].startswith('test/') and f['path'].endswith('.wav')]
    def audio(item):
        path=item['path'];expected=item.get('lfs',{}).get('oid')
        if expected in known:return None
        dest=ROOT/'data/raw/external/audio'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            for attempt in range(5):
                try:dest.write_bytes(urllib.request.urlopen(base+path,timeout=60).read());break
                except Exception:
                    if attempt==4:raise
                    time.sleep(2)
        h=digest(dest)
        if expected and h!=expected:raise RuntimeError('HF LFS hash mismatch: '+path)
        return dict(id='external_audio_'+path.replace('/','_'),modality='audio',label=int('/fake/' in path),path=dest.relative_to(ROOT).as_posix(),group=path,source=base+path,sha256=h,split='external_test')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(audio,files),1):
            if r is not None and r['sha256'] not in known:rows.append(r);known.add(r['sha256'])
            else:skipped.append(files[i-1]['path'])
            if i%50==0:print('External audio processed',i,flush=True)
    dump('data/external_manifest.json',rows)
    dump('results/external_data_audit.json',dict(total=len(rows),text=sum(r['modality']=='text' for r in rows),audio=sum(r['modality']=='audio' for r in rows),excluded_exact_duplicates=len(skipped),notes=['Audio remains same corpus as course; expansion, not cross-corpus evaluation.','Text is different source/model generation benchmark; models and thresholds frozen.','No speaker identity guarantee or near-duplicate semantic audit.']))
    print('External data complete:',len(rows),flush=True)
