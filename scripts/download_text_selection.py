"""Additional development validation plus untouched final test after diagnostic shift findings."""
import sys,urllib.request,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *

if __name__=='__main__':
    known={r['sha256'] for r in read('data/manifest.json')+read('data/external_manifest.json')};rows=[]
    for source_split,target_split,start in [('valid','validation_external',0),('test','final_external_test',100)]:
        for kind,label in [('webtext',0),('small-117M-k40',1)]:
            url=f'https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/{kind}.{source_split}.jsonl';count=0
            with urllib.request.urlopen(url,timeout=60) as response:
                for index,line in enumerate(response):
                    if index<start:continue
                    text=json.loads(line)['text'];data=text.encode('utf-8');h=hashlib.sha256(data).hexdigest()
                    if h in known:continue
                    path=ROOT/f'data/raw/text_selection/{target_split}/{kind}_{index:04d}.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);known.add(h)
                    rows.append(dict(id=f'{target_split}_{kind}_{index}',modality='text',label=label,path=path.relative_to(ROOT).as_posix(),group=f'{kind}_{source_split}_{index}',source=url,sha256=h,split=target_split))
                    count+=1
                    if count==100:break
            print(target_split,kind,count,flush=True)
    dump('data/text_selection_manifest.json',rows)
    dump('results/text_selection_protocol.json',dict(reason='Course text baseline failed on initial source-shift diagnostic set. Added independent external development validation and NEW final test examples.',
         diagnostic='original external_manifest text indices 0..99 per source; results observed before protocol extension',
         validation='official valid split, first 100 unique examples per class',final_test='official test split, starting at index100, 100 new unique examples per class',
         selection='equal weight of course validation balanced accuracy and external validation balanced accuracy; tie baseline',
         caveat='Exploratory protocol extension after diagnostic results; final test is separate, not a preregistered study'))
