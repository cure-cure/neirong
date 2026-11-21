"""Migration-specific regressions; CUDA execution itself must be checked on AutoDL."""
import sys,tempfile,unittest,importlib.util,json,runpy,subprocess,os
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class MigrationTests(unittest.TestCase):
    def test_workspace_keeps_snapshot_and_starts_without_scores(self):
        spec=importlib.util.spec_from_file_location('workspace',ROOT/'scripts/create_gpu_workspace.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            parent=Path(temporary).resolve();self.assertTrue(parent.is_relative_to(ROOT));source=parent/'snapshot';source.mkdir()
            for name in ['src','scripts','tests','configs','vendor','models','data','references','results','reports']:(source/name).mkdir()
            (source/'models/weight.bin').write_bytes(b'unchanged weight');(source/'results/old.json').write_text('CPU cache')
            (source/'src/__pycache__').mkdir();(source/'src/__pycache__/ignored.pyc').write_bytes(b'cache')
            (source/'app.py').write_text('# fixture')
            target=module.create_workspace(source,parent/'gpu')
            self.assertEqual((target/'models/weight.bin').read_bytes(),b'unchanged weight')
            self.assertEqual(list((target/'results').iterdir()),[])
            self.assertFalse((target/'src/__pycache__').exists())
            self.assertTrue((source/'results/old.json').exists())
            with self.assertRaises(ValueError):module.create_workspace(source,source/'nested')
            with self.assertRaises(FileExistsError):module.create_workspace(source,target)

    def test_no_face_is_recorded_without_fabricated_attack(self):
        import torch
        from src import common
        class DummyDetector:
            def __init__(self,_):self.model=torch.nn.Linear(1,1)
        def fake_read(path):
            if str(path)=='data/manifest.json':return [dict(id='video_missingface',modality='video',split='test',label=0,group='v0',path='missing.mp4')]
            if str(path)=='models/video_calibration.json':return dict(paper=dict(slope=1,intercept=0))
            return []
        with patch('src.paper_models.PaperDetector',DummyDetector),patch('src.engine.load_value',side_effect=ValueError('No face')),patch.object(common,'read',side_effect=fake_read),patch.object(common,'dump') as output,patch.object(sys,'argv',['run_attacks.py','--modality','video','--profile','full']):
            runpy.run_path(str(ROOT/'scripts/run_attacks.py'),run_name='__main__')
        summary=next(c.args[1] for c in output.call_args_list if str(c.args[0]).endswith('_summary.json'))
        self.assertEqual(summary['requested_samples'],1);self.assertEqual(len(summary['input_failures']),1)
        self.assertTrue(all(r['n']==0 and r['asr'] is None for r in summary['results']))

    def test_cuda_preflight_refuses_cpu_configuration(self):
        env=dict(os.environ,LAB_DEVICE='cpu',PYTHONIOENCODING='utf-8')
        result=subprocess.run([sys.executable,str(ROOT/'scripts/gpu_preflight.py'),'--require-cuda','--skip-hashes'],env=env,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        self.assertNotEqual(result.returncode,0);self.assertIn('CUDA required',result.stdout+result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
