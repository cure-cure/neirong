import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import read,digest,ROOT

class IntegrityTests(unittest.TestCase):
    def test_source_hashes_and_counts(self):
        rows=read('data/manifest.json');self.assertEqual(len(rows),348)
        for row in rows:self.assertEqual(digest(ROOT/row['path']),row['sha256'])
    def test_groups_and_duplicates_do_not_cross_splits(self):
        rows=read('data/manifest.json');groups={};hashes={}
        for r in rows:
            groups.setdefault(r['group'],set()).add(r['split']);hashes.setdefault(r['sha256'],set()).add(r['split'])
        self.assertTrue(all(len(v)==1 for v in groups.values()))
        self.assertTrue(all(len(v)==1 for v in hashes.values()))
    def test_attacks_change_loss_without_breaking_budget(self):
        import torch
        from src.attacks import attack
        x=torch.full((2,3,4,4),.5)
        fn=lambda z:z.flatten(1).sum(1)-24
        adv=attack(fn,x,0,.03,steps=5,restarts=2)
        self.assertLessEqual(float((adv-x).abs().max()),.030001)
        self.assertTrue(torch.all(fn(adv)>fn(x)))
        self.assertTrue(torch.all((adv>=0)&(adv<=1)))
    def test_calibration_does_not_invert_model_direction(self):
        from src.metrics import calibrate,probability
        c=calibrate([-2,-1,1,2],[1,1,0,0]);self.assertGreater(c['slope'],0)
        self.assertLess(float(probability(-1,c)),float(probability(1,c)))
    def test_available_paper_weights_are_strictly_loaded(self):
        from src.paper_models import PaperDetector
        model=PaperDetector('audio');self.assertFalse(model.model.training);model.close()
    def test_differentiable_audio_matches_independent_scipy_pipeline(self):
        import numpy as np,torch
        from src.audio_baseline_torch import AudioBaselineTorch
        from src.engine import Engine,load_value
        rows=[r for r in read('data/manifest.json') if r['modality']=='audio']
        module=AudioBaselineTorch();engine=Engine()
        for row in [rows[0],rows[-1]]:
            wave=load_value('audio',ROOT/row['path']);expected=engine.baseline_score('audio',wave)
            actual=float(module(torch.tensor(wave)[None]).item())
            self.assertAlmostEqual(expected,actual,places=3)
    def test_unsupported_language_is_marked_uncertain(self):
        from src.engine import Engine
        result=Engine().infer('text','这是一段用于检查中文输入是否被明确标记为超出英语验证范围的文字。'*3,'baseline')
        self.assertEqual(result['judgement'],'存疑');self.assertTrue(result['quality_warnings'])
    def test_silent_audio_is_not_classified_as_real(self):
        import tempfile,numpy as np,soundfile as sf
        from src.media import load_audio
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            self.assertTrue(Path(temporary).resolve().is_relative_to(ROOT.resolve()))
            path=Path(temporary)/'silence.wav';sf.write(path,np.zeros(16000),16000)
            with self.assertRaisesRegex(ValueError,'静音'):load_audio(path)

if __name__=='__main__':unittest.main(verbosity=2)
