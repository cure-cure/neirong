"""Differentiable equivalent of the trained SciPy LFCC baseline for white-box audits."""
import torch,joblib,numpy as np
from torch import nn
from scipy.fft import dct
from .common import ROOT

class AudioBaselineTorch(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__();model=joblib.load(ROOT/'models/audio_baseline.joblib');scaler,lr=model
        bins=np.linspace(0,256,26);freq=np.arange(257)
        filt=np.array([np.maximum(0,np.minimum((freq-bins[i])/(bins[i+1]-bins[i]),(bins[i+2]-freq)/(bins[i+2]-bins[i+1]))) for i in range(24)])
        for name,data in dict(filters=filt,dct_matrix=dct(np.eye(24),type=2,axis=0,norm='ortho')[:20],
                              mean=scaler.mean_,scale=scaler.scale_,weight=lr.coef_[0],bias=lr.intercept_).items():
            self.register_buffer(name,torch.tensor(data,dtype=torch.float32))
        self.register_buffer('window',torch.hann_window(400));self.to(device).eval()
    def forward(self,x):
        if x.shape[-1]<400:x=torch.nn.functional.pad(x,(0,400-x.shape[-1]))
        n=x.shape[-1];padded=400+int(np.ceil((n-400)/160))*160
        x=torch.nn.functional.pad(x,(0,padded-n));frames=x.unfold(-1,400,160)*self.window
        spectrum=torch.fft.rfft(frames,n=512,dim=-1)/self.window.sum()
        power=spectrum.abs().square().transpose(1,2)
        cep=self.dct_matrix@torch.log(self.filters@power+1e-10)
        delta=cep[:,:,1:]-cep[:,:,:-1]
        change=delta.abs().mean(-1) if delta.shape[-1] else torch.zeros_like(cep[:,:,0])
        features=torch.cat([cep.mean(-1),cep.std(-1,correction=0),change],dim=-1)
        return ((features-self.mean)/self.scale)@self.weight+self.bias[0]
