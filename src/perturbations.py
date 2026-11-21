import io,re
import numpy as np
from PIL import Image,ImageFilter,ImageEnhance
from scipy import signal

CONDITIONS={
 'text':[('spaces',1),('typos',.01),('truncate',.75),('truncate',.5),('punctuation',1)],
 'image':[('jpeg',90),('jpeg',50),('resize',.5),('resize',.25),('blur',1),('blur',2),('noise',.01),('brightness',.7)],
 'audio':[('noise',30),('noise',20),('noise',10),('gain',.5),('resample',8000),('truncate',.75),('speed',1.05)],
 'video':[('jpeg',50),('resize',.5),('blur',1),('brightness',.7)]}

def transform(modality,value,name,level,seed=202430):
    rng=np.random.default_rng(seed)
    if modality=='text':
        if name=='spaces':return re.sub(r' ', '  ',value)
        if name=='punctuation':return value.replace('"','').replace('“','').replace('”','')
        if name=='truncate':return value[:int(len(value)*level)]
        chars=list(value);eligible=[i for i,c in enumerate(chars) if c.isascii() and c.isalpha()]
        for i in rng.choice(eligible,min(len(eligible),max(1,int(len(eligible)*level))),replace=False):chars[i]=chr((ord(chars[i].lower())-97+1)%26+97)
        return ''.join(chars)
    if modality=='image':
        if name=='jpeg':
            b=io.BytesIO();value.save(b,format='JPEG',quality=int(level));b.seek(0);return Image.open(b).convert('RGB')
        if name=='resize':return value.resize((max(1,int(value.width*level)),max(1,int(value.height*level))),Image.Resampling.BILINEAR).resize(value.size,Image.Resampling.BILINEAR)
        if name=='blur':return value.filter(ImageFilter.GaussianBlur(level))
        if name=='brightness':return ImageEnhance.Brightness(value).enhance(level)
        a=np.asarray(value)/255;return Image.fromarray(np.uint8(np.clip(a+rng.normal(0,level,a.shape),0,1)*255))
    if modality=='audio':
        if name=='noise':
            rms=np.sqrt(np.mean(value**2));return np.clip(value+rng.normal(0,rms*10**(-level/20),value.shape),-1,1).astype(np.float32)
        if name=='gain':return (value*level).astype(np.float32)
        if name=='truncate':return value[:max(400,int(len(value)*level))]
        if name=='resample':return signal.resample_poly(signal.resample_poly(value,int(level),16000),16000,int(level)).astype(np.float32)
        if name=='speed':return signal.resample(value,max(400,int(len(value)/level))).astype(np.float32)
    if modality=='video':
        return [{**f,'face':transform('image',f['face'],name,level,seed)} for f in value]
    raise ValueError((modality,name))

