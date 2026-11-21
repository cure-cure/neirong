"""Sequential CUDA stages, logged and isolated from local baseline results."""
import sys,argparse,subprocess,datetime,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(arguments,log):
    command=[sys.executable,*arguments];print('RUN',command,flush=True)
    log.write('\nRUN '+repr(command)+'\n');log.flush()
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
    for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    code=process.wait()
    if code:raise subprocess.CalledProcessError(code,command)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reproduce','robustness','attacks','report']);args=p.parse_args()
    if not (ROOT/'GPU_WORKSPACE.json').exists():raise SystemExit('Run inside a fresh workspace created by create_gpu_workspace.py, not the local baseline snapshot.')
    import os,torch
    if args.stage!='report' and (os.environ.get('LAB_DEVICE')!='cuda' or not torch.cuda.is_available()):raise SystemExit('Set LAB_DEVICE=cuda; a working CUDA PyTorch is required.')
    tasks=[]
    if args.stage=='reproduce':
        tasks=[['scripts/run_experiments.py','--method','all']]
        for m in ['text','audio']:
            for method in ['baseline','paper']:tasks.append(['scripts/evaluate_external.py','--modality',m,'--method',method])
        for method in ['baseline','paper']:tasks.append(['scripts/evaluate_external.py','--modality','text','--method',method,'--manifest','data/text_selection_manifest.json','--split','validation_external','--name','external_validation'])
        tasks.append(['scripts/select_methods.py'])
        for method in ['baseline','paper']:tasks.append(['scripts/evaluate_external.py','--modality','text','--method',method,'--manifest','data/text_selection_manifest.json','--split','final_external_test','--name','final_external'])
    elif args.stage=='robustness':
        tasks=[['scripts/run_robustness.py','--method',method] for method in ['baseline','paper','combined']]
    elif args.stage=='attacks':
        tasks=[['scripts/run_attacks.py','--modality',m,'--method',method,'--profile','full'] for m,method in [('image','paper'),('video','paper'),('audio','baseline'),('audio','paper')]]
    else:
        tasks=[['-m','unittest','discover','-s','tests','-v'],['scripts/shortcut_audit.py'],['scripts/verify_audio_roundtrip.py'],['scripts/final_audit.py'],['scripts/build_report.py']]
    directory=ROOT/'results/gpu_logs';directory.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');log_path=directory/f'{args.stage}_{stamp}.log'
    with log_path.open('w',encoding='utf-8') as log:
        for task in tasks:run(task,log)
    print('STAGE COMPLETE',args.stage,log_path,flush=True)
