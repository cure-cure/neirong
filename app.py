"""Local UI. No model inference calls external APIs; uploaded files stay local."""
import os,time,uuid,json,sys,subprocess,threading,hashlib
from pathlib import Path
import re,urllib.parse
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ['NO_PROXY']=os.environ.get('NO_PROXY','')+',127.0.0.1,localhost,::1'
os.environ['no_proxy']=os.environ['NO_PROXY']
import gradio as gr
import pandas as pd
import numpy as np
from src.common import ROOT,read,dump,MODALITIES
from src.engine import Engine,load_value
from src.evidence import evidence

ENGINE=Engine();LOCK=threading.Lock()
METHODS={'验证集选定的默认方法':'selected','基础方法':'baseline','论文方法':'paper','基础＋论文等权融合':'combined'}
MODES={v:k for k,v in MODALITIES.items()}

def report_for_ui():
    path=ROOT/'reports/实验报告.md'
    if not path.exists():return '报告将根据实际结果生成。'
    content=path.read_text('utf-8')
    return re.sub(r'\]\(\.\./(results/[^)]+)\)',lambda m:'](/gradio_api/file='+urllib.parse.quote((ROOT/m.group(1)).as_posix(),safe='/:')+')',content)

def unable_assessment(mode,reason):
    run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    result=dict(run_id=run_id,modality=mode,judgement='存疑／无法充分评估',confidence=0.,fake_probability=None,
                authenticity_score=None,risk='无法评估',evidence=[reason],process=['输入／检测检查未通过'],
                note='无法评估不等于真实或伪造，不参与联合评分。')
    path=f'results/ui_runs/{run_id}.json';dump(path,result)
    return '### 无法充分评估\n\n'+reason+'\n\n置信度：0；真实性：未评分；风险：无法评估。',result,None,[],str(ROOT/path),None

def run_detection(modality,method,text,file,progress=gr.Progress()):
    mode=MODES[modality]
    if mode=='text' and text.strip():value=text.strip();source='typed_text'
    else:
        if not file:return unable_assessment(mode,'请上传对应模态文件，或输入文本。')
        source=str(file);progress(.1,desc='解码与预处理')
        try:value=load_value(mode,file)
        except Exception as e:return unable_assessment(mode,str(e))
    if mode=='text' and len(value.strip())<20:return unable_assessment(mode,'文本过短，无法提供可靠检测。')
    with LOCK:
        progress(.3,desc='加载模型并进行真实推理（首次加载较慢）')
        try:result=ENGINE.infer(mode,value,METHODS[method])
        except Exception as e:return unable_assessment(mode,str(e))
        progress(.85,desc='生成检测证据与四层结果');fig,gallery=evidence(mode,value,result,ENGINE.paper if mode=='image' and 'paper' in result['components'] else None)
    run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6];result['run_id']=run_id;result['source']=source
    result['input_sha256']=hashlib.sha256(value.encode('utf-8')).hexdigest() if mode=='text' else hashlib.sha256(Path(source).read_bytes()).hexdigest()
    evidence_dir=ROOT/'results/ui_runs'/run_id;evidence_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(evidence_dir/'evidence.png',dpi=140)
    result['evidence_files']=[dict(path=(evidence_dir/'evidence.png').relative_to(ROOT).as_posix(),caption='特征／检测过程图')]
    for index,(im,caption) in enumerate(gallery):
        image_path=evidence_dir/f'image_{index}.png';im.save(image_path)
        result['evidence_files'].append(dict(path=image_path.relative_to(ROOT).as_posix(),caption=caption))
    result['process']=['输入质量检查','模态预处理','基础特征／论文模型推理','验证集概率校准','方法融合与冲突检查','四层评估输出']
    path=f'results/ui_runs/{run_id}.json';dump(path,result)
    brief=f'### {result["judgement"]}\n\n伪造概率 **{result["fake_probability"]:.3f}**　判定置信度 **{result["confidence"]:.3f}**\n\n真实性评分 **{result["authenticity_score"]:.3f}**　风险等级 **{result["risk"]}**　推理耗时 **{result["seconds"]:.2f}s**\n\n置信度来自小规模验证集校准；跨域与中文表现需另行验证。频谱与特征贡献为辅助线索，不能证明篡改位置。'
    import matplotlib.pyplot as plt
    plt.close(fig)
    progress(1,desc='完成');return brief,result,fig,gallery,str(ROOT/path),result

def metrics_table():
    rows=[]
    for p in sorted((ROOT/'results').glob('*_metrics.json')):
        obj=read(p);r=dict(modality=MODALITIES[obj['modality']],method=obj['method']);r.update(obj['test']);r.pop('confusion_matrix',None);r['coverage']=obj.get('coverage');r['test_coverage']=obj.get('test_coverage');r['effective_accuracy']=obj.get('effective_accuracy_counting_failures_as_wrong');rows.append(r)
    return pd.DataFrame(rows)

def errors_table(modality,method,only_errors):
    actual=METHODS[method]
    if actual=='selected':actual=read('configs/selected_methods.json')[MODES[modality]]['selected']
    p=ROOT/f'results/{MODES[modality]}_{actual}_scores.json'
    if not p.exists():return pd.DataFrame()
    rows=[]
    for r in read(p):
        if r.get('split')!='test':continue
        if 'error' in r:rows.append(dict(id=r['id'],status='处理失败',error=r['error']));continue
        pred=int(r['fake_probability']>=.5);wrong=pred!=r['label']
        if only_errors and not wrong:continue
        rows.append(dict(id=r['id'],label=r['label'],prediction=pred,fake_probability=r['fake_probability'],error_type='误报' if wrong and pred else '漏报' if wrong else '正确',path=r['path']))
    return pd.DataFrame(rows)

def inspect_result_detail(modality,method,only_errors):
    import matplotlib.pyplot as plt
    actual=METHODS[method];m=MODES[modality]
    if actual=='selected':actual=read('configs/selected_methods.json')[m]['selected']
    p=ROOT/f'results/{m}_{actual}_metrics.json';fig,ax=plt.subplots(figsize=(5,4))
    if p.exists():
        obj=read(p);cm=np.asarray(obj['test']['confusion_matrix']);ax.imshow(cm,cmap='Blues')
        for (i,j),number in np.ndenumerate(cm):ax.text(j,i,str(number),ha='center',va='center',fontsize=18)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],xlabel='Predicted',ylabel='Ground truth',title=m+' / '+actual+' / held-out test')
    fig.tight_layout();plt.close(fig);return errors_table(modality,method,only_errors),fig

def external_table():
    rows=[]
    for pattern in ['external_*_summary.json','final_external_*_summary.json']:
        for p in sorted((ROOT/'results').glob(pattern)):
            obj=read(p)
            if 'metrics' in obj:rows.append(dict(experiment=p.stem,**obj['metrics']))
    return pd.DataFrame(rows)

def experiment_records(kind):
    import matplotlib.pyplot as plt
    patterns={'普通鲁棒性':'robustness_*_summary.json','对抗自评':'attacks_*_summary.json'}
    objects={p.stem:read(p) for p in sorted((ROOT/'results').glob(patterns[kind]))};rows=[]
    for name,obj in objects.items():
        for r in obj.get('results',[]):rows.append(dict(experiment=name,**r))
    fig,ax=plt.subplots(figsize=(12,4))
    if kind=='对抗自评':
        for name,obj in objects.items():
            for defense in [False,True]:
                subset=[r for r in obj.get('results',[]) if r.get('defense')==defense and r.get('asr') is not None and r['attack'].startswith('PGD')]
                if subset:ax.plot([r['epsilon'] for r in subset],[r['asr'] for r in subset],marker='o',label=name+(' / smooth' if defense else ' / clean detector'))
        ax.set(xlabel='L-infinity epsilon',ylabel='Attack success rate (conditional on clean correctness)',ylim=(-.05,1.05))
        if ax.lines:ax.legend(fontsize=6)
    else:
        labels=[];values=[]
        for name,obj in objects.items():
            for r in obj.get('results',[]):labels.append(name.replace('robustness_','')+'/'+r['condition']+str(r['level']));values.append(r['accuracy'])
        ax.bar(np.arange(len(values)),values);ax.set(xticks=np.arange(len(labels)),xticklabels=labels,ylim=(0,1.05),ylabel='Held-out accuracy')
        ax.tick_params(axis='x',labelrotation=90,labelsize=5)
    fig.tight_layout();plt.close(fig);return objects,pd.DataFrame(rows),fig

def fuse(a,b,c,d):
    supplied=[r for r in [a,b,c,d] if isinstance(r,dict) and 'fake_probability' in r]
    if not supplied:raise gr.Error('先完成至少一个模态的现场检测。')
    scores=[r['fake_probability'] for r in supplied];risk=float(np.mean(scores));maximum=max(scores)
    return dict(components=[dict(modality=r['modality'],fake_probability=r['fake_probability'],run_id=r.get('run_id'),input_sha256=r.get('input_sha256')) for r in supplied],
                fusion_rule='等权加权融合；缺失模态不参与，剩余权重重新归一化',weighted_fake_score=risk,
                component_max_risk=maximum,vote_fake_fraction=sum(p>=.5 for p in scores)/len(scores),
                authenticity_score=1-risk,confidence=max(risk,1-risk),confidence_note='融合判定强度，尚未经过联合概率校准',risk='高' if maximum>=.7 else '低' if maximum<=.3 else '中',judgement='存疑' if .35<risk<.65 or max(scores)-min(scores)>.5 else '整体疑似AI生成／篡改' if risk>=.65 else '整体为真实倾向',
                note='融合分数不是经验证的联合概率。单个可疑组成部分可能被平均掩盖，需同时查看最高分项风险。仅合并同一关联材料；不判断语义一致性或事实真伪。')

def attack_demo(modality,method,file,label,epsilon,steps,defended):
    import torch
    from src.attacks import attack
    from src.metrics import probability
    import matplotlib.pyplot as plt
    m=MODES[modality]
    if not file:raise gr.Error('请上传图像或音频。')
    actual=METHODS[method]
    if actual=='selected':actual=read('configs/selected_methods.json')[m]['selected']
    if actual=='combined':raise gr.Error('攻击演示请选择基础或论文检测器。')
    if actual=='baseline' and m!='audio':raise gr.Error('基础方法白盒演示目前支持音频；图像请选择论文检测器。')
    with LOCK:
        if actual=='baseline':
            import types
            from src.audio_baseline_torch import AudioBaselineTorch
            from src.common import torch_setup
            dev=torch_setup();module=AudioBaselineTorch(dev)
            model=types.SimpleNamespace(model=module,audio_tensor=lambda a:torch.tensor(a,dtype=torch.float32,device=dev)[None],audio_logit=module)
        else:model=ENGINE.get_paper(m)
        for param in model.model.parameters():param.requires_grad_(False)
        if hasattr(model,'head'):
            for param in model.head.parameters():param.requires_grad_(False)
        value=load_value(m,file);x=model.audio_tensor(value) if m=='audio' else model.image_tensor(value)
        def fn(z):
            if defended:
                z=torch.nn.functional.avg_pool1d(torch.nn.functional.pad(z[:,None],(1,1),mode='reflect'),3,1)[:,0] if m=='audio' else torch.nn.functional.avg_pool2d(torch.nn.functional.pad(z,(1,1,1,1),mode='reflect'),3,1)
            return model.audio_logit(z) if m=='audio' else model.image_logit(z)
        cal=read(f'models/{m}_calibration.json')[actual]
        with torch.no_grad():before=float(probability(fn(x).item(),cal))
        start=time.perf_counter();adv=attack(fn,x,int(label),float(epsilon),int(steps),1,int(steps)>1,-1 if m=='audio' else 0,1)
        with torch.no_grad():after=float(probability(fn(adv).item(),cal))
        delta=(adv-x).detach().cpu().numpy();original=x.detach().cpu().numpy();changed=adv.cpu().numpy()
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    if m=='image':
        axes[0].imshow(original[0].transpose(1,2,0));axes[1].imshow(changed[0].transpose(1,2,0));axes[2].imshow(np.abs(delta[0]).mean(0),cmap='magma')
    else:
        axes[0].plot(original[0]);axes[1].plot(changed[0]);axes[2].plot(delta[0])
    for ax,title in zip(axes,['Original model input','Adversarial model input','Perturbation (display scale differs)']):ax.set_title(title)
    fig.tight_layout();r=dict(method=actual,clean_fake_probability=before,adv_fake_probability=after,epsilon=epsilon,actual_linf=float(np.max(np.abs(delta))),
                             steps=int(steps),defense=defended,adaptive=defended,seconds=time.perf_counter()-start,
                             attack_success=bool(int(before>=.5)==int(label) and int(after>=.5)!=int(label)),
                             scope='完整音频波形' if actual=='baseline' else '首个4.04秒音频片段／图像模型输入裁剪；不代表文件级全流程攻击')
    run_id=uuid.uuid4().hex[:10];dump(f'results/ui_attacks/{run_id}.json',r)
    preview=None
    if m=='audio':
        import soundfile as sf
        path=ROOT/f'results/ui_attacks/{run_id}.wav';sf.write(path,changed[0],16000,subtype='PCM_16');preview=str(path)
    plt.close(fig);return r,fig,preview

CSS='''#title {padding:18px 0;border-bottom:2px solid #3b82f6} .gradio-container {max-width:1400px!important}'''
with gr.Blocks(title='实验六 · 四模态真实性检测',css=CSS) as app:
    gr.Markdown('# 四模态内容真实性检测与自评\n胡逸辰 · 2024302181004 · 基础方法与论文方法对照',elem_id='title')
    gr.Markdown('上传内容 → 预处理 → 真实模型推理 → 概率校准 → 证据 → 综合评估。模型在本地运行；批量实验与现场检测分别记录。')
    states=[]
    with gr.Tabs():
        for modality in ['文本','图像','音频','视频']:
            with gr.Tab(modality+'检测'):
                with gr.Row():
                    with gr.Column(scale=1):
                        method=gr.Radio(list(METHODS),value='验证集选定的默认方法',label='检测方法')
                        text=gr.Textbox(label='输入英文文本（课程文本集为英语）',lines=7,visible=modality=='文本')
                        file=gr.File(label='上传文件',type='filepath',file_count='single')
                        button=gr.Button('开始检测并生成证据',variant='primary')
                        sample_rows=read('data/manifest.json')
                        samples=[r for r in sample_rows if r['modality']==MODES[modality] and r['split']=='test']
                        examples=[]
                        for label in [0,1]:
                            row=next(r for r in samples if r['label']==label);path=ROOT/row['path']
                            examples.append([path.read_text('utf-8-sig'),'验证集选定的默认方法'] if modality=='文本' else [str(path),'验证集选定的默认方法'])
                        gr.Examples(examples=examples,inputs=[text,method] if modality=='文本' else [file,method],label='课程留出样本示例（真实／伪造，点击载入）')
                        gr.Markdown('视频当前检测人脸伪造，采样 8 帧，逐帧取最大人脸。无可用人脸时报告无法评估。' if modality=='视频' else '“存疑”包括低置信度和方法强冲突。')
                    with gr.Column(scale=2):
                        brief=gr.Markdown();plot=gr.Plot();gallery=gr.Gallery(label='输入／检测过程',columns=4,height=230)
                        with gr.Accordion('四层结果、模型依据与运行过程',open=False):raw=gr.JSON()
                        export=gr.File(label='下载本次检测记录');state=gr.State();states.append(state)
                button.click(run_detection,[gr.State(modality),method,text,file],[brief,raw,plot,gallery,export,state],concurrency_limit=1)
        with gr.Tab('关联材料联合评估'):
            gr.Markdown('先在各检测页处理属于同一组材料的输入，再做等权加权融合，同时保留最高分项风险与投票结果。尚未进行语义一致性验证。')
            fusion=gr.JSON();gr.Button('汇总当前四模态结果').click(fuse,states,fusion)
        with gr.Tab('测试总览与误报漏报'):
            table=gr.Dataframe(value=metrics_table(),interactive=False,label='独立留出测试集指标（不包含训练／验证样本）')
            gr.Button('刷新批量测试结果').click(metrics_table,outputs=table)
            with gr.Row():em=gr.Dropdown(list(MODES),value='图像',label='模态');et=gr.Dropdown(list(METHODS),value='基础＋论文等权融合',label='方法');only=gr.Checkbox(value=True,label='仅显示误报／漏报')
            cmplot=gr.Plot();errors=gr.Dataframe(interactive=False);gr.Button('查看混淆矩阵与测试样本明细').click(inspect_result_detail,[em,et,only],[errors,cmplot])
            extra=gr.Dataframe(value=external_table(),interactive=False,label='外部诊断、独立验证与新保留测试（按实验名称区分）');gr.Button('刷新扩展评估').click(external_table,outputs=extra)
            gr.JSON(value=read('configs/selected_methods.json'),label='默认方法选择依据（只使用验证数据）')
            gr.JSON(value=read('results/data_audit.json'),label='数据来源、划分与已知限制')
        with gr.Tab('鲁棒性与对抗自评'):
            kind=gr.Radio(['普通鲁棒性','对抗自评'],value='普通鲁棒性',label='批量实验记录');record_table=gr.Dataframe(interactive=False,label='已运行实验统计');record_plot=gr.Plot();records=gr.JSON()
            gr.Button('读取已运行的实验记录').click(experiment_records,kind,[records,record_table,record_plot])
            gr.Markdown('### 单样本白盒攻击演示\nFGSM：1 步、无随机初始化；PGD：多步、随机初始化。开启平滑防御后，攻击穿过防御重新计算梯度。')
            with gr.Row():am=gr.Dropdown(['音频','图像'],value='音频',label='攻击模块');amd=gr.Dropdown(['验证集选定的默认方法','基础方法','论文方法'],value='验证集选定的默认方法',label='攻击检测器');af=gr.File(type='filepath',label='上传样本');al=gr.Radio([0,1],value=0,label='已知标签：0真实／1伪造')
            with gr.Row():ae=gr.Number(value=.001,label='L∞ 扰动上限（图像可用 0.007843）');ast=gr.Slider(1,20,value=5,step=1,label='攻击步数');ad=gr.Checkbox(label='平滑防御＋自适应攻击')
            ao=gr.JSON();ap=gr.Plot();aa=gr.Audio(label='对抗音频试听（16kHz PCM，图像攻击不生成音频）');gr.Button('执行真实攻击（CPU 可能较慢）',variant='primary').click(attack_demo,[am,amd,af,al,ae,ast,ad],[ao,ap,aa],concurrency_limit=1)
        with gr.Tab('报告与方法说明'):
            report=ROOT/'reports/实验报告.md';report_view=gr.Markdown(report_for_ui())
            gr.Button('刷新最新实验报告').click(report_for_ui,outputs=report_view)
            gr.File(value=[str(p) for p in [ROOT/'reports/实验报告.md',ROOT/'README.md'] if p.exists()],label='下载报告与使用说明')

if __name__=='__main__':app.queue(default_concurrency_limit=1,max_size=16).launch(server_name='127.0.0.1',server_port=int(os.environ.get('LAB_PORT','7860')),inbrowser=False,share=False,allowed_paths=[str(ROOT/'results'),str(ROOT/'reports')])
