"""Remove only our verified redundant download chunks, never model weights or data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read,digest

if __name__=='__main__':
    sources={r['path']:r for r in read('models/sources.json')};removed=0
    for relative in ['models/distilgpt2/model.safetensors','models/ViT-L-14.pt']:
        final=(ROOT/relative).resolve();assert final.is_relative_to((ROOT/'models').resolve())
        assert final.exists() and digest(final)==sources[relative]['sha256']
        chunks=final.parent/(final.name+'.chunks')
        if chunks.exists():
            assert chunks.resolve().is_relative_to((ROOT/'models').resolve())
            print('Verified temporary target:',chunks,flush=True)
            for part in chunks.iterdir():
                if part.is_file() and part.name.isdigit():removed+=part.stat().st_size;part.unlink()
            chunks.rmdir()
        partial=final.with_suffix(final.suffix+'.part')
        if partial.exists():removed+=partial.stat().st_size;partial.unlink()
    print('Removed redundant cache bytes:',removed,flush=True)
