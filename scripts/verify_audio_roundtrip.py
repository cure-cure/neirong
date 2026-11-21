import sys,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import soundfile as sf,numpy as np
from src.common import *
from src.engine import Engine
from src.metrics import probability

if __name__=='__main__':
    rows=read('results/attacks_audio_baseline_full.json');engine=Engine();cal=read('models/audio_calibration.json')['baseline'];results=[]
    for row in rows:
        if row['defense']:continue
        path=ROOT/f'results/attack_examples/audio_baseline_{row["id"]}_{row["attack"]}_{row["epsilon"]}_{row["defense"]}.npz'
        data=np.load(path);wave=data['adversarial'][0];buffer=io.BytesIO();sf.write(buffer,wave,16000,format='WAV',subtype='PCM_16');buffer.seek(0);decoded,sr=sf.read(buffer,dtype='float32')
        raw=engine.baseline_score('audio',decoded);score=float(probability(raw,cal))
        results.append(dict(id=row['id'],attack=row['attack'],epsilon=row['epsilon'],label=row['label'],clean_correct=row['clean_correct'],adv_correct_after_pcm=int(score>=.5)==row['label'],fake_probability_after_pcm=score,
                            actual_linf_after_pcm=float(np.max(np.abs(decoded-data['clean'][0]))),note='PCM16 quantization can add up to one quantization step beyond floating-point perturbation budget'))
    dump('results/audio_pcm_roundtrip.json',results)
    print('PCM16 round-trip verified:',len(results),'attacks; errors:',sum(not r['adv_correct_after_pcm'] for r in results),flush=True)
