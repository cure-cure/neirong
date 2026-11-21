from pathlib import Path
import json, os, hashlib, random, uuid
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEED=202430
MODALITIES={'text':'文本','image':'图像','audio':'音频','video':'视频'}
def dump(path,obj):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,p)
def read(path):return json.loads((ROOT/path).read_text('utf-8'))
def seed():
    random.seed(SEED);np.random.seed(SEED)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def device():
    import torch
    requested=os.environ.get('LAB_DEVICE','cpu')
    if requested=='auto':return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    return requested
def torch_setup():
    import torch
    torch.manual_seed(SEED);torch.set_num_threads(int(os.environ.get('LAB_THREADS','4')))
    return device()
