"""Research detectors with explicit checkpoint provenance and local-only loading."""
import importlib.util,gc,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from .common import ROOT,torch_setup

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(ROOT/path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class PaperDetector:
    def __init__(self,modality):
        self.modality=modality;self.device=torch_setup();self.model=None
        if modality=='text':
            from transformers import AutoTokenizer,AutoModelForCausalLM
            self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/distilgpt2',local_files_only=True)
            self.model=AutoModelForCausalLM.from_pretrained(ROOT/'models/distilgpt2',local_files_only=True).to(self.device).eval()
            self.description='Fast-DetectGPT analytic conditional curvature; DistilGPT2 same scoring/reference model, 256-token cap. Adapted lightweight configuration, not original paper benchmark.'
        elif modality=='image':
            m=module('univfd_clip','vendor/univfd/model.py')
            with (ROOT/'models/ViT-L-14.pt').open('rb') as stream:
                archive=torch.jit.load(stream,map_location='cpu')
            state={k.removeprefix('visual.'):v for k,v in archive.state_dict().items() if k.startswith('visual.')}
            self.model=m.VisionTransformer(input_resolution=224,patch_size=14,width=1024,layers=24,heads=16,output_dim=768)
            self.model.load_state_dict(state,strict=True)
            del archive,state;gc.collect();self.model=self.model.float().to(self.device).eval()
            self.head=nn.Linear(768,1)
            self.head.load_state_dict(torch.load(ROOT/'models/univfd_fc.pth',map_location='cpu',weights_only=True),strict=True)
            self.head=self.head.to(self.device).eval()
            self.description='UnivFD CVPR2023; official CLIP ViT-L/14 + official ProGAN-trained linear head; adapted CLIP resize-shorter-side224 + center crop (official repository validation uses center-crop-only).'
        elif modality=='audio':
            m=module('aasist','vendor/aasist/models/AASIST.py')
            cfg=json.loads((ROOT/'vendor/aasist/config/AASIST-L.conf').read_text())['model_config']
            self.model=m.Model(cfg).to(self.device)
            self.model.load_state_dict(torch.load(ROOT/'vendor/aasist/models/weights/AASIST-L.pth',map_location=self.device,weights_only=True),strict=True)
            self.model.eval();self.description='Official AASIST-L ASVspoof2019 LA checkpoint; 16kHz, 64600-sample chunks, repeat-pad short audio; class 0=spoof.'
        elif modality=='video':
            m=module('deepfake_xception','vendor/deepfakebench/xception.py')
            self.model=m.Xception(dict(mode='original',num_classes=2,inc=3,dropout=False))
            weights=torch.load(ROOT/'models/xception_best.pth',map_location='cpu',weights_only=True)
            if 'state_dict' in weights:weights=weights['state_dict']
            weights={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in weights.items()}
            self.model.load_state_dict(weights,strict=True);self.model=self.model.to(self.device).eval()
            self.description='FaceForensics++ Xception, DeepfakeBench official trained detector checkpoint; 256x256 face crops, RGB normalized to [-1,1]; mean frame probability.'
        else:raise ValueError(modality)
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        if hasattr(self,'head'):
            for parameter in self.head.parameters():parameter.requires_grad_(False)

    def image_tensor(self,im):
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode
        if self.modality=='image':
            return TF.to_tensor(TF.center_crop(TF.resize(im,224,interpolation=InterpolationMode.BICUBIC),[224,224])).unsqueeze(0).to(self.device)
        return TF.to_tensor(im.resize((256,256),Image.Resampling.BILINEAR)).unsqueeze(0).to(self.device)

    def image_logit(self,x):
        if self.modality=='image':
            mean=x.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
            std=x.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
            return self.head(self.model((x-mean)/std)).reshape(-1)
        logits=self.model((x-.5)/.5)
        if isinstance(logits,tuple):logits=logits[0]
        return logits[:,1]-logits[:,0]

    def audio_tensor(self,a):
        if len(a)<64600:a=np.tile(a,int(np.ceil(64600/len(a))))
        return torch.tensor(np.array(a[:64600],copy=True),dtype=torch.float32,device=self.device).unsqueeze(0)

    def audio_logit(self,x):
        _,logits=self.model(x);return logits[:,0]-logits[:,1]

    @torch.inference_mode()
    def score(self,value):
        if self.modality=='text':
            tokens=self.tokenizer(value,return_tensors='pt',max_length=256,truncation=True).to(self.device)
            ids=tokens['input_ids'];n=ids.shape[1]
            if n<8:raise ValueError('文本过短：至少需要 8 个模型 token')
            logits=self.model(**tokens).logits[:,:-1,:].float()
            logp=logits.log_softmax(-1);p=logp.exp();observed=logp.gather(-1,ids[:,1:,None]).squeeze(-1)
            expectation=(p*logp).sum(-1);variance=(p*logp.square()).sum(-1)-expectation.square()
            raw=((observed-expectation).sum()/variance.sum().clamp_min(1e-8).sqrt()).item()
            return raw,dict(tokens_scored=n-1,truncated=len(self.tokenizer.encode(value,truncation=True,max_length=257))>256,
                            token_logp=observed[0].cpu().tolist(),curvature=float(raw))
        if self.modality=='image':
            return float(self.image_logit(self.image_tensor(value)).item()),{}
        if self.modality=='audio':
            logits=[];starts=list(range(0,len(value),64600))
            for start in starts:logits.append(float(self.audio_logit(self.audio_tensor(value[start:start+64600])).item()))
            probs=1/(1+np.exp(-np.clip(logits,-40,40)));p=float(np.mean(probs));raw=float(np.log((p+1e-8)/(1-p+1e-8)))
            return raw,dict(segment_seconds=[i/16000 for i in starts],segment_fake_scores=probs.tolist())
        logits=[float(self.image_logit(self.image_tensor(f['face'])).item()) for f in value]
        p=1/(1+np.exp(-np.clip(logits,-40,40)));mean=float(p.mean())
        return float(np.log((mean+1e-8)/(1-mean+1e-8))),dict(frame_fake_scores=p.tolist(),seconds=[f['seconds'] for f in value])

    def close(self):
        self.model=None
        if hasattr(self,'head'):self.head=None
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
