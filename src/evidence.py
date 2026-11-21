import numpy as np,joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from .common import ROOT

def evidence(modality,value,result,model=None):
    fig,axes=plt.subplots(1,2,figsize=(12,4));gallery=[];details=result.get('details',{})
    if modality=='text':
        model=joblib.load(ROOT/'models/text_baseline.joblib');vec=model[0];x=vec.transform([value]);contribution=x.toarray()[0]*model[-1].coef_[0]
        indices=np.argsort(np.abs(contribution))[-12:];words=vec.get_feature_names_out()[indices]
        axes[0].barh(words,contribution[indices]);axes[0].set_title('TF-IDF baseline contributions (not proof)')
        lp=details.get('token_logp',[])
        axes[1].plot(lp if lp else [len(s.split()) for s in value.splitlines() if s.strip()]);axes[1].set_title('Token log likelihood' if lp else 'Words per non-empty line')
    elif modality=='image':
        a=np.asarray(value);axes[0].imshow(a);axes[0].axis('off');axes[0].set_title('Decoded RGB image')
        spectrum=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(value.resize((256,256))).mean(2)))))
        axes[1].imshow(spectrum,cmap='magma');axes[1].set_title('Fourier spectrum (auxiliary cue)');axes[1].axis('off');gallery=[(value,'原始图像')]
        if model is not None:
            import torch
            from PIL import Image
            x=model.image_tensor(value).detach().requires_grad_(True)
            gradient=torch.autograd.grad(model.image_logit(x).sum(),x)[0]
            saliency=gradient.abs().mean(1)[0].detach().cpu().numpy();saliency=saliency/(np.quantile(saliency,.99)+1e-8)
            heat=plt.get_cmap('magma')(np.clip(saliency,0,1))[:,:,:3]
            crop=x[0].detach().cpu().numpy().transpose(1,2,0)
            overlay=Image.fromarray(np.uint8(np.clip(.55*crop+.45*heat,0,1)*255))
            gallery.extend([(Image.fromarray(np.uint8(crop*255)),'实际224×224模型输入'),(overlay,'输入梯度敏感度；不是真实篡改区域')])
    elif modality=='audio':
        axes[0].plot(np.arange(len(value))/16000,value,linewidth=.5);axes[0].set_title('Waveform / seconds')
        f,t,z=signal.stft(value,16000,nperseg=min(512,len(value)));axes[1].pcolormesh(t,f,20*np.log10(np.abs(z)+1e-7),shading='auto',cmap='magma');axes[1].set_title('STFT spectrogram / Hz')
    else:
        p=details.get('frame_fake_scores',[]);times=details.get('seconds',list(range(len(value))))
        axes[0].plot(times,p if p else [0]*len(value),marker='o');axes[0].set_ylim(0,1);axes[0].set_title('Frame fake scores (raw checkpoint)')
        axes[1].imshow(value[0]['face']);axes[1].axis('off');axes[1].set_title('First detected face crop')
        gallery=[(f['face'],f'{f["seconds"]:.2f}s / frame {f["index"]}') for f in value if f['seconds'] is not None]
    fig.tight_layout();return fig,gallery
