"""Deterministic spectral/MFCC feature extraction shared by training and inference."""
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft
from scipy.fft import dct
SR=32000
SIZE=SR*4

def load_audio(path):
    y,sr=sf.read(str(path),dtype='float32',always_2d=True)
    y=y.mean(axis=1)
    if sr!=SR:
        g=math.gcd(int(sr),SR);y=resample_poly(y,SR//g,int(sr)//g)
    y=np.nan_to_num(y).astype(np.float32)
    if len(y)<SR//2 or np.max(np.abs(y))<1e-7:raise ValueError('0.5초 미만이거나 무음인 파일')
    return y

def windows(y,training=False):
    if len(y)<=SIZE:return [np.pad(y,(0,SIZE-len(y)))]
    if training:
        starts=np.linspace(0,len(y)-SIZE,min(10,math.ceil(len(y)/SIZE))).astype(int)
    else:
        starts=list(range(0,len(y)-SIZE+1,SIZE//2))
        if starts[-1]!=len(y)-SIZE:starts.append(len(y)-SIZE)
    return [y[s:s+SIZE] for s in starts]

def _melbank():
    hz=lambda m:700*(10**(m/2595)-1)
    mel=lambda h:2595*np.log10(1+h/700)
    knots=hz(np.linspace(mel(40),mel(SR/2),66))
    freq=np.fft.rfftfreq(1024,1/SR)
    bank=[]
    for a,b,c in zip(knots[:-2],knots[1:-1],knots[2:]):
        t=np.maximum(0,np.minimum((freq-a)/(b-a),(c-freq)/(c-b)))
        bank.append(t/max(t.sum(),1e-12))
    return np.asarray(bank,dtype=np.float32)
BANK=_melbank()

def features(y):
    y=np.asarray(y,dtype=np.float32)
    y=y/max(float(np.max(np.abs(y))),1e-6)
    _,_,z=stft(y,fs=SR,nperseg=1024,noverlap=512,boundary=None)
    power=np.abs(z)**2
    lm=10*np.log10(np.maximum(BANK@power,1e-12))
    lm=lm-np.max(lm)
    mf=dct(lm,type=2,axis=0,norm='ortho')[:40]
    delta=np.diff(lm,axis=1)
    env=np.sqrt(np.mean(power,axis=0))
    env=(env-env.mean())/max(float(env.std()),1e-7)
    rhythm=[np.mean(env[:-lag]*env[lag:]) if len(env)>lag else 0 for lag in [1,2,3,4,6,8,12,16,24,32,48,64]]
    v=np.concatenate([lm.mean(1),lm.std(1),np.quantile(lm,.1,axis=1),np.quantile(lm,.9,axis=1),mf.mean(1),mf.std(1),np.abs(delta).mean(1),rhythm])
    return np.nan_to_num(v).astype(np.float32)
