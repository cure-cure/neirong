"""Create a fresh run directory: never reuse local score caches as CUDA evidence."""
from pathlib import Path
import argparse,shutil,json,datetime

ROOT=Path(__file__).resolve().parents[1]
def create_workspace(root,destination):
    root=Path(root).resolve();target=Path(destination).expanduser().resolve()
    if target==root or root.is_relative_to(target) or target.is_relative_to(root):
        raise ValueError('Destination must be a separate sibling directory, not project root or its parent/child.')
    target.mkdir(parents=True,exist_ok=False)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.part','*.chunks','.venv')
    for name in ['src','scripts','tests','configs','vendor','models','data','references']:
        shutil.copytree(root/name,target/name,ignore=ignore)
    for path in root.iterdir():
        if path.is_file() and path.suffix in ['.py','.md','.txt','.sh','.ps1']:shutil.copy2(path,target/path.name)
    (target/'results').mkdir();(target/'reports').mkdir()
    info=dict(parent_snapshot=str(root),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fresh_results=True,
              note='Models/calibration/data copied; results and reports intentionally start empty. Do not rerun prepare_data or download data.')
    (target/'GPU_WORKSPACE.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True);args=parser.parse_args()
    print(create_workspace(ROOT,args.destination))
