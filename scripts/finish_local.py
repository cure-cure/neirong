"""Sequential heavy jobs: keep at most one deep experiment resident at a time."""
import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(*arguments):
    print('\nRUN',*arguments,flush=True)
    subprocess.run([sys.executable,*arguments],cwd=ROOT,check=True)
if __name__=='__main__':
    while not (ROOT/'results/video_paper_metrics.json').exists():
        print('Waiting for existing video evaluation to finish; no additional heavy model loaded.',flush=True);time.sleep(30)
    run('scripts/run_experiments.py','--modality','video','--method','combined')
    run('scripts/select_methods.py')
    for method in ['baseline','paper']:
        run('scripts/evaluate_external.py','--modality','text','--method',method,'--manifest','data/text_selection_manifest.json','--split','final_external_test','--name','final_external')
    run('scripts/run_robustness.py','--modality','image','--method','paper')
    run('scripts/run_robustness.py','--modality','video','--method','paper')
    run('scripts/run_attacks.py','--modality','image','--profile','local','--limit','2')
    run('scripts/run_attacks.py','--modality','video','--profile','local','--limit','2')
    run('scripts/verify_audio_roundtrip.py')
    run('-m','unittest','discover','-s','tests','-v')
    run('scripts/build_report.py')
    run('scripts/final_audit.py')
    print('LOCAL EXPERIMENT PIPELINE COMPLETE',flush=True)
