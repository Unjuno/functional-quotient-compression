"""Leakage-controlled logMel acoustic features, deterministic file selection."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal
import librosa
import torch

SOURCE = '/opt/pyvenv/lib/python3.13/site-packages/pyannote/audio/sample/sample.wav'
SECONDARY = '/opt/pyvenv/lib/python3.13/site-packages/gradio/media_assets/audio/audio_sample.wav'
SOURCE_SHA='c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e'
SEGMENTS={'train':(0,17),'dev':(18,21),'fresh':(22,25),'late':(26,29)}

def load_wave(path=SOURCE):
    y,sr=sf.read(path,always_2d=True)
    y=y.astype(np.float32).mean(axis=1)
    if sr!=16000:
        from math import gcd
        g=gcd(sr,16000)
        y=scipy.signal.resample_poly(y,16000//g,sr//g).astype(np.float32)
    return y

def features(y:np.ndarray):
    # torch STFT keeps exact center=False frame alignment, 25 ms / 10 ms hop.
    x=torch.from_numpy(y.copy())
    ft=torch.stft(x,n_fft=400,hop_length=160,win_length=400,window=torch.hann_window(400),center=False,return_complex=True)
    power=ft.abs().square()
    mel=torch.tensor(librosa.filters.mel(sr=16000,n_fft=400,n_mels=64,fmin=80,fmax=7600),dtype=torch.float32)
    return torch.log1p(10*(mel@power).T)

def main_data():
    h=hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()
    assert h==SOURCE_SHA,('audio asset changed',h)
    full=features(load_wave())
    # A window starting at t spans [t,t+25ms); 1s empty gaps prevent border leakage.
    idx=torch.arange(full.shape[0])*0.01
    sets={s: full[(idx>=a)&(idx+0.025<=b)] for s,(a,b) in SEGMENTS.items()}
    mu=sets['train'].mean(0)
    sd=sets['train'].std(0).clamp_min(1e-4)
    norm={s:(v-mu)/sd for s,v in sets.items()}
    assert len(norm['train'])>1500 and len(norm['dev'])>200
    return norm,mu,sd,{'frames':{k:len(v) for k,v in norm.items()},'audio_sha256':h,'raw_shape':list(full.shape)}

def cross_data(mu,sd):
    return (features(load_wave(SECONDARY))-mu)/sd
