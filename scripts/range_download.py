"""Resumable parallel byte ranges for official large model artifacts."""
import urllib.request,concurrent.futures,os,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def download(url,path,workers=6):
    dest=ROOT/path;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():print('Exists',path,flush=True);return
    probe=urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=45)
    cr=probe.headers.get('Content-Range');probe.close()
    if not cr:raise RuntimeError('Server does not support byte ranges')
    total=int(cr.split('/')[-1]);size=4*1024*1024;parts=dest.parent/(dest.name+'.chunks');parts.mkdir(exist_ok=True)
    def chunk(i):
        start=i*size;end=min(total-1,start+size-1);target=parts/str(i)
        if target.exists() and target.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                # Cache busting prevents a cached first-byte response from shadowing ranges.
                u=url+('&' if '?' in url else '?')+f'lab_chunk={i}'
                req=urllib.request.Request(u,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as src:
                    actual=src.headers.get('Content-Range','')
                    if not actual.startswith(f'bytes {start}-{end}/'):raise RuntimeError(f'Unexpected range: {actual}')
                    data=src.read()
                if len(data)!=end-start+1:raise RuntimeError('Incomplete chunk')
                target.write_bytes(data);return
            except Exception:
                if attempt==4:raise
                time.sleep(1)
    n=(total+size-1)//size;print('Downloading',path,total,'bytes,',n,'chunks',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for count,_ in enumerate(pool.map(chunk,range(n)),1):
            if count%10==0:print(path,count,'/',n,flush=True)
    with dest.with_suffix(dest.suffix+'.joined').open('wb') as f:
        for i in range(n):f.write((parts/str(i)).read_bytes())
    dest.with_suffix(dest.suffix+'.joined').replace(dest)
    print('Complete',path,hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':
    download('https://huggingface.co/distilbert/distilgpt2/resolve/main/model.safetensors','models/distilgpt2/model.safetensors')
    download('https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt','models/ViT-L-14.pt')
