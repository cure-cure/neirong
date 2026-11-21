"""End-to-end tests via the running Gradio HTTP upload/queue/export endpoints."""
import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['NO_PROXY']='127.0.0.1,localhost';os.environ['no_proxy']=os.environ['NO_PROXY']
from gradio_client import Client,handle_file
from src.common import *

if __name__=='__main__':
    client=Client('http://127.0.0.1:7860',verbose=False);rows=read('data/manifest.json');results=[]
    for m,endpoint in [('text','/run_detection'),('image','/run_detection_1'),('audio','/run_detection_2'),('video','/run_detection_3')]:
        row=next(r for r in rows if r['modality']==m and r['split']=='test');path=ROOT/row['path']
        result=client.predict('验证集选定的默认方法',path.read_text('utf-8-sig') if m=='text' else '',None if m=='text' else handle_file(path),api_name=endpoint)
        data=result[1]
        assert all(k in data for k in ['judgement','confidence','authenticity_score','risk','components','process'])
        assert result[2]['type']=='matplotlib'
        assert Path(result[4]).exists()
        results.append(dict(modality=m,run_id=data['run_id'],method=data['method'],judgement=data['judgement'],probability=data['fake_probability'],plot=True,export=True))
        print('UI PASS',m,data['method'],data['judgement'],flush=True)
    fused=client.predict(api_name='/fuse');assert len(fused['components'])==4
    unavailable=client.predict('基础方法','短文本',None,api_name='/run_detection')
    assert unavailable[1]['confidence']==0 and unavailable[1]['authenticity_score'] is None
    attack=next(r for r in rows if r['modality']=='audio' and r['split']=='test')
    result=client.predict('音频','基础方法',handle_file(ROOT/attack['path']),attack['label'],.001,5,False,api_name='/attack_demo')
    assert result[0]['actual_linf']<=.001001
    dump('results/ui_verification.json',dict(detections=results,fusion=fused,attack=result[0],unavailable_input_handled=True,fusion_test_scope='Functional check with unrelated course samples; NOT a paired semantic or crossmodal accuracy benchmark',visual_inspection='Not performed: computer-use kernel initialization failed; HTTP/API checks are not visual layout verification'))
    print('UI fusion and attack PASS',flush=True)
