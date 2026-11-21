"""Build three portable archives with CRC verification and a per-file SHA256 inventory."""
from pathlib import Path
import zipfile,hashlib,json,datetime,shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'AutoDL上传包_20261118'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        while block:=source.read(1024*1024):h.update(block)
    return h.hexdigest()
def valid(path):return not any(p in ['__pycache__','.venv','.git'] for p in path.parts) and path.suffix not in ['.pyc','.part'] and not any(p.endswith('.chunks') for p in path.parts)
def files(directory):return [p for p in (ROOT/directory).rglob('*') if p.is_file() and valid(p.relative_to(ROOT))]

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    code=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.txt','.ps1']]
    for directory in ['src','scripts','tests','configs','references','vendor']:code+=files(directory)
    aasist_weights=[p for p in code if p.suffix=='.pth' and 'weights' in p.parts]
    code=[p for p in code if p not in aasist_weights]
    assets=files('data')+files('models')+aasist_weights
    baseline=files('results')+files('reports')
    plans=[('01','lab6_01_code_and_handoff.zip',code),('02','lab6_02_data_and_models.zip',assets),('03','lab6_03_local_results.zip',baseline)]
    total=sum(p.stat().st_size for _,_,items in plans for p in items)
    if shutil.disk_usage(OUT).free<total+100*1024**2:raise SystemExit('Insufficient temporary disk space for safe packaging.')
    inventory=dict(created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),root_in_archives='lab6',gpu_executed=False,packages=[])
    all_names=set()
    for number,name,items in plans:
        target=OUT/name
        if target.exists():raise SystemExit('Archive already exists; preserve it and choose a new version directory: '+str(target))
        entries=[];items=sorted(set(items),key=lambda p:p.relative_to(ROOT).as_posix())
        for p in items:
            relative=p.relative_to(ROOT).as_posix();assert relative not in all_names;all_names.add(relative)
            entries.append(dict(path=relative,bytes=p.stat().st_size,sha256=sha(p)))
        meta=dict(package=name,files=len(items),uncompressed_bytes=sum(e['bytes'] for e in entries),entries=entries)
        print('Packaging',name,len(items),'files',meta['uncompressed_bytes'],'bytes',flush=True)
        partial=target.with_suffix('.zip.part')
        with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for index,p in enumerate(items,1):
                relative=p.relative_to(ROOT).as_posix()
                compressed=p.suffix.lower() not in ['.npz','.png','.jpg','.jpeg','.mp4']
                archive.write(p,'lab6/'+relative,compress_type=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED)
                if index%250==0:print(name,index,'/',len(items),flush=True)
            archive.writestr('lab6/PACKAGE_META_'+number+'.json',json.dumps(meta,ensure_ascii=False,indent=2))
        partial.replace(target)
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
            assert len(archive.infolist())==len(items)+1
            assert all(n.startswith('lab6/') and '..' not in Path(n).parts for n in archive.namelist())
            for entry in entries:
                h=hashlib.sha256()
                with archive.open('lab6/'+entry['path']) as stream:
                    while block:=stream.read(1024*1024):h.update(block)
                assert h.hexdigest()==entry['sha256'], 'Snapshot changed while packaging: '+entry['path']
        meta.update(zip_bytes=target.stat().st_size,zip_sha256=sha(target),crc_verified=True);inventory['packages'].append(meta)
        print('Verified',name,meta['zip_bytes'],'bytes',flush=True)
    (OUT/'PACKAGE_CONTENTS.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'SHA256SUMS.txt').write_text(''.join(p['zip_sha256']+'  '+p['package']+'\n' for p in inventory['packages']),encoding='utf-8',newline='\n')
    for name in ['AUTODL_HANDOFF.md','AUTODL_AGENT_PROMPT.md']:shutil.copy2(ROOT/name,OUT/name)
    (OUT/'先读我.txt').write_text('请上传三个lab6_*.zip和SHA256SUMS.txt到AutoDL。三个包都解压到同一父目录，内部统一为lab6/。先阅读AUTODL_HANDOFF.md；给AI助手接手时同时提供AUTODL_AGENT_PROMPT.md。不要上传Windows .venv。当前CUDA任务尚未在服务器执行。\n',encoding='utf-8')
    print('PACKAGING COMPLETE',OUT,flush=True)
