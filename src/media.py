from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from scipy import signal
from scipy.fft import dct
from .common import ROOT

def load_image(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((2048,2048));return im.copy()

def image_features(im):
    a=np.asarray(im.resize((128,128)),dtype=np.float32)/255
    gray=a.mean(2);f=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray-gray.mean()))))
    yy,xx=np.indices(f.shape);rr=np.sqrt((yy-64)**2+(xx-64)**2)
    radial=[float(f[(rr>=i*5)&(rr<(i+1)*5)].mean()) for i in range(16)]
    gx=np.diff(gray,axis=0);gy=np.diff(gray,axis=1)
    return np.array(radial+list(a.mean((0,1)))+list(a.std((0,1)))+
                    [float(np.abs(gx).mean()),float(np.abs(gy).mean()),float(gx.std()),float(gy.std())],dtype=np.float32)

def load_audio(path,sr=16000):
    import soundfile as sf
    a,rate=sf.read(str(path),dtype='float32',always_2d=True)
    if len(a)==0:raise ValueError('音频为空')
    a=a.mean(axis=1)
    if rate!=sr:
        from math import gcd
        g=gcd(rate,sr);a=signal.resample_poly(a,sr//g,rate//g).astype(np.float32)
    if not np.isfinite(a).all():raise ValueError('音频包含非有限数值')
    if len(a)<400:raise ValueError('音频过短，不足25毫秒')
    if float(np.sqrt(np.mean(a**2)))<1e-7:raise ValueError('静音或近静音，无法评估语音真实性')
    return a,sr

def audio_features(a,sr=16000):
    if len(a)<400:a=np.pad(a,(0,400-len(a)))
    _,_,z=signal.stft(a,fs=sr,nperseg=400,noverlap=240,nfft=512,boundary=None)
    power=np.abs(z)**2
    # Linear filterbank cepstra, not MFCC: 24 equal-frequency triangular bands.
    bins=np.linspace(0,len(power)-1,26)
    filt=np.array([np.maximum(0,np.minimum((np.arange(len(power))-bins[i])/(bins[i+1]-bins[i]),(bins[i+2]-np.arange(len(power)))/(bins[i+2]-bins[i+1]))) for i in range(24)])
    cep=dct(np.log(filt@power+1e-10),type=2,axis=0,norm='ortho')[:20]
    delta=np.diff(cep,axis=1)
    return np.concatenate([cep.mean(1),cep.std(1),np.abs(delta).mean(1) if delta.shape[1] else np.zeros(20)]).astype(np.float32)

def video_frames(path,count=8):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        if n<1:raise ValueError('视频无法解码或无帧')
        cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        frames=[]
        for index in np.unique(np.linspace(0,n-1,min(n,count)).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,bgr=cap.read()
            if not ok:continue
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            boxes=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(40,40))
            if len(boxes)==0:continue
            x,y,w,h=max(boxes,key=lambda b:int(b[2])*int(b[3]));margin=int(max(w,h)*.2)
            x0,y0=max(0,x-margin),max(0,y-margin);x1,y1=min(rgb.shape[1],x+w+margin),min(rgb.shape[0],y+h+margin)
            frames.append(dict(index=int(index),seconds=float(index/fps) if fps>0 else None,
                               face=Image.fromarray(rgb[y0:y1,x0:x1]),frame=Image.fromarray(rgb),box=[int(x),int(y),int(w),int(h)]))
        if not frames:raise ValueError('采样帧中未检测到人脸，不能进行人脸伪造评估')
        return frames,dict(total_frames=n,fps=fps,sampled=count,faces=len(frames),face_detector='OpenCV Haar; largest face per frame; no identity tracking')
    finally:cap.release()

def video_features(frames):
    features=np.stack([image_features(f['face']) for f in frames]);return np.concatenate([features.mean(0),features.std(0)])

