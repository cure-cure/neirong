import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import read,digest,ROOT

def requires_assets(*paths):
    return unittest.skipUnless(
        all((ROOT/path).exists() for path in paths),'requires excluded model/data assets')

class IntegrityTests(unittest.TestCase):
    @requires_assets('data/raw/course')
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
    @requires_assets('vendor/aasist/models/weights/AASIST-L.pth')
    def test_available_paper_weights_are_strictly_loaded(self):
        from src.paper_models import PaperDetector
        model=PaperDetector('audio');self.assertFalse(model.model.training);model.close()
    @requires_assets('models/audio_baseline.joblib','data/raw/course')
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
    @requires_assets('models/text_calibration.json','models/text_baseline.joblib')
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

    @requires_assets('data/raw/external/cifake_test')
    def test_external_visual_manifest_is_fixed_balanced_and_hashed(self):
        rows=read('data/external_visual_manifest.json')
        self.assertEqual(len(rows),200);self.assertEqual({label:sum(r['label']==label for r in rows) for label in [0,1]},{0:100,1:100})
        self.assertTrue(all(r['split']=='external_test' and r['width']==32 and r['height']==32 for r in rows))
        self.assertTrue(all(digest(ROOT/r['path'])==r['sha256'] for r in rows))
        course={r['sha256'] for r in read('data/manifest.json')}
        self.assertFalse(course & {r['sha256'] for r in rows})

    def test_full_visual_attack_coverage_and_budgets(self):
        expected={'image':(480,20,20,0),'video':(456,19,20,1)}
        for modality,(records,effective,requested,failures) in expected.items():
            rows=read(f'results/attacks_{modality}_paper_full.json')
            summary=read(f'results/attacks_{modality}_paper_full_summary.json')
            self.assertEqual(len(rows),records);self.assertEqual(len({r['id'] for r in rows}),effective)
            self.assertEqual(summary['requested_samples'],requested);self.assertEqual(len(summary['input_failures']),failures)
            self.assertEqual(len(summary['results']),24)
            self.assertTrue(all(r['n']==effective for r in summary['results']))
            self.assertTrue(all(r['linf']<=r['epsilon']+1e-5 for r in rows))

if __name__=='__main__':unittest.main(verbosity=2)
