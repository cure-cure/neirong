import time,joblib
from pathlib import Path
import numpy as np
from .common import ROOT,read,dump
from .media import *
from .metrics import probability

def load_value(modality,path):
    if modality=='text':return Path(path).read_text('utf-8-sig')
    if modality=='image':return load_image(path)
    if modality=='audio':return load_audio(path)[0]
    if modality=='video':return video_frames(path)[0]
    raise ValueError(modality)

def features(modality,value):
    if modality=='text':return value
    if modality=='image':return image_features(value)
    if modality=='audio':return audio_features(value)
    return video_features(value)

def feature_names(modality):
    image=[f'FFT_ring_{i}' for i in range(16)]+[f'RGB_mean_{c}' for c in 'RGB']+[f'RGB_std_{c}' for c in 'RGB']+['mean_abs_dy','mean_abs_dx','std_dy','std_dx']
    if modality=='image':return image
    if modality=='audio':return [f'LFCC_{kind}_{i}' for kind in ['mean','std','delta_abs_mean'] for i in range(20)]
    return [f'frame_{kind}_{name}' for kind in ['mean','std'] for name in image]

class Engine:
    def __init__(self):self.paper=None;self.current=None
    def get_paper(self,modality):
        if self.current!=modality or self.paper is None:
            if self.paper:self.paper.close();self.paper=None
            self.current=None
            from .paper_models import PaperDetector
            self.paper=PaperDetector(modality);self.current=modality
        return self.paper
    def baseline_score(self,modality,value,return_details=False):
        model=joblib.load(ROOT/f'models/{modality}_baseline.joblib')
        f=features(modality,value)
        raw=float(model.decision_function([f] if modality=='text' else np.asarray(f)[None])[0])
        if not return_details:return raw
        if modality=='text':
            transformed=model[0].transform([f]);contribution=transformed.toarray()[0]*model[-1].coef_[0]
            index=np.argsort(np.abs(contribution))[-12:]
            details=dict(nonzero_tfidf_features=int(transformed.nnz),top_features=[dict(term=str(model[0].get_feature_names_out()[i]),logit_contribution=float(contribution[i])) for i in index])
        else:
            normalized=model[0].transform(np.asarray(f)[None])[0]
            contribution=normalized*model[-1].coef_[0]
            details=dict(feature_names=feature_names(modality),feature_vector=np.asarray(f).tolist(),standardized_features=normalized.tolist(),logit_contributions=contribution.tolist(),intercept=float(model[-1].intercept_[0]))
        return raw,details
    def infer(self,modality,value,method='combined'):
        requested_method=method;selection_reason=None
        if method=='selected':
            selected=read('configs/selected_methods.json')[modality];method=selected['selected']
            selection_reason=selected.get('reason',selected['rule'])
        if method=='baseline' and self.current!=modality and self.paper is not None:
            self.paper.close();self.paper=None;self.current=None
        started=time.perf_counter();cal=read(f'models/{modality}_calibration.json');components={};details={}
        if method in ['baseline','combined']:
            raw,feature_details=self.baseline_score(modality,value,True);details['baseline_features']=feature_details
            components['baseline']=dict(raw=raw,fake_probability=float(probability(raw,cal['baseline'])),calibration=cal['baseline'])
        if method in ['paper','combined']:
            paper=self.get_paper(modality);raw,paper_details=paper.score(value);details.update(paper_details)
            components['paper']=dict(raw=raw,fake_probability=float(probability(raw,cal['paper'])),description=paper.description,calibration=cal['paper'])
        p=float(np.mean([x['fake_probability'] for x in components.values()]));conf=max(p,1-p)
        doubtful=conf<.65 or (len(components)>1 and abs(components['paper']['fake_probability']-components['baseline']['fake_probability'])>.5)
        warnings=[]
        if modality=='text' and sum('\u4e00'<=c<='\u9fff' for c in value)/max(1,len(value))>.15:
            warnings.append('中文超出当前英语验证范围；分数仅供探索');doubtful=True
        if modality=='image' and min(value.size)<64:
            warnings.append('图像分辨率过低，证据不足');doubtful=True
        label='存疑' if doubtful else ('AI生成／篡改' if p>=.5 else '真实倾向')
        return dict(modality=modality,method=method,requested_method=requested_method,selection_reason=selection_reason,judgement=label,binary_prediction=int(p>=.5),fake_probability=p,
                    confidence=conf,confidence_note='验证集校准后的估计；小样本、跨域条件下不保证可靠',
                    authenticity_score=1-p,risk='高' if p>=.7 else '低' if p<=.3 else '中',
                    components=components,details=details,quality_warnings=warnings,seconds=time.perf_counter()-started,
                    limitations=['来源／生成检测不等于事实核查','证据可视化仅表示模型线索，不构成篡改定位真值'])

