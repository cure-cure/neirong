"""Create a fixed, balanced CIFAKE external image test subset.

The input is the publisher dataset's test Parquet mirrored at a pinned Hugging
Face commit. Selection is deterministic and does not inspect model outputs.
"""
import argparse,hashlib,io,datetime
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,digest,dump,read

UPSTREAM='https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images'
MIRROR='https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset'
COMMIT='6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31'
DOI='10.1109/ACCESS.2024.3356122'

def rank(index):
    return hashlib.sha256(f'cifake-test:{index}'.encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parquet',type=Path,required=True)
    parser.add_argument('--per-class',type=int,default=100)
    args=parser.parse_args()
    source=args.parquet.resolve()
    table=pq.read_table(source,columns=['image','label'])
    labels=table.column('label').to_pylist()
    if len(labels)!=20000 or {int(x) for x in labels}!={0,1}:
        raise ValueError(f'Expected the 20,000-row CIFAKE test split, got {len(labels)} rows')
    selected=[]
    for source_label in [0,1]:
        candidates=[i for i,value in enumerate(labels) if int(value)==source_label]
        selected.extend(sorted(candidates,key=rank)[:args.per_class])
    image_values=table.column('image').to_pylist()
    course_hashes={row['sha256'] for row in read('data/manifest.json')}
    output=ROOT/'data/raw/external/cifake_test'
    manifest=[];overlaps=[]
    for index in sorted(selected):
        source_label=int(labels[index]);project_label=1 if source_label==0 else 0
        value=image_values[index]
        blob=value.get('bytes') if isinstance(value,dict) else None
        if not blob:raise ValueError(f'Missing encoded image bytes at row {index}')
        with Image.open(io.BytesIO(blob)) as image:
            image.load();width,height=image.size
            suffix='.jpg' if image.format in {'JPEG','JPG'} else '.png'
        category='fake' if project_label else 'real'
        target=output/category/f'cifake_test_{index:05d}{suffix}'
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
        sha=digest(target)
        if sha in course_hashes:overlaps.append(index)
        manifest.append(dict(
            id=f'cifake_{category}_{index:05d}',modality='image',label=project_label,
            path=target.relative_to(ROOT).as_posix(),group=f'cifake_test_{index:05d}',
            split='external_test',source='CIFAKE official test split',source_row=index,
            generator='Stable Diffusion 1.4' if project_label else None,
            real_source='CIFAR-10' if not project_label else None,
            width=width,height=height,bytes=len(blob),sha256=sha))
    if overlaps:raise ValueError(f'Exact byte overlap with course data: {overlaps}')
    dump('data/external_visual_manifest.json',manifest)
    dump('data/external_visual_source.json',dict(
        dataset='CIFAKE: Real and AI-Generated Synthetic Images',paper_doi=DOI,
        upstream=UPSTREAM,mirror=MIRROR,mirror_commit=COMMIT,
        source_file_sha256=digest(source),source_rows=len(labels),selected_rows=len(manifest),
        selection=f'{args.per_class} per class by smallest SHA256(cifake-test:<row index>); model outputs never inspected',
        label_mapping={'CIFAKE 0 FAKE':1,'CIFAKE 1 REAL':0},
        content='REAL: CIFAR-10; FAKE: Stable Diffusion 1.4 equivalents',
        resolution='32x32 RGB; evaluated through the project unchanged preprocessing pipeline',
        license_note='Dataset publisher states MIT; underlying CIFAR-10 source terms and attribution still apply.',
        exact_course_byte_overlaps=0,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(f'Prepared {len(manifest)} external images ({args.per_class} per class); no course byte overlaps')

if __name__=='__main__':main()
