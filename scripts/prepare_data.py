import sys,re,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from sklearn.model_selection import GroupShuffleSplit

def main():
    rows=[];seen={};duplicates=[]
    for modality,folder in MODALITIES.items():
        items=[]
        for name,label in [('real',0),('fake',1)]:
            for path in sorted((ROOT/'data/raw/course'/folder/name).glob('*')):
                h=digest(path)
                if h in seen:duplicates.append([str(path),seen[h]])
                seen[h]=str(path)
                stem=path.stem
                group=re.search(r'\d+',stem).group() if modality in ['text','video'] else stem
                items.append(dict(id=f'{modality}_{name}_{stem}',modality=modality,label=label,
                                  path=path.relative_to(ROOT).as_posix(),group=modality+'_'+group,
                                  source='course_share',sha256=h,bytes=path.stat().st_size))
        groups=np.array([i['group'] for i in items]);indices=np.arange(len(items))
        train,rest=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=SEED).split(indices,groups=groups))
        val_ix,test_ix=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=SEED+1).split(rest,groups=groups[rest]))
        mapping={**{int(i):'train' for i in train},**{int(rest[i]):'val' for i in val_ix},**{int(rest[i]):'test' for i in test_ix}}
        for i,row in enumerate(items):row['split']=mapping[i]
        rows+=items
    dump('data/manifest.json',rows)
    summary={m:{s:dict(collections.Counter(str(r['label']) for r in rows if r['modality']==m and r['split']==s)) for s in ['train','val','test']} for m in MODALITIES}
    dump('results/data_audit.json',dict(total=len(rows),splits=summary,exact_duplicates=duplicates,
         limitations=['Audio speaker/source identities not provided: speaker-disjoint split cannot be established.',
         'Video original/fake pairs grouped by numeric ID; identity-disjoint status unknown.',
         'Text numeric pairs grouped conservatively; source model IDs unavailable in course subset.',
         'Public pretrained checkpoint overlap with course samples cannot be fully ruled out.',
         'No independently collected personal dataset has been supplied or recorded.']))
    print(summary)
if __name__=='__main__':main()
