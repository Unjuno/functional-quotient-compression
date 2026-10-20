"""Separate exact PCM lossless reference, never conflated with Mel autoencoder."""
import io,json,hashlib
from pathlib import Path
import soundfile as sf
import numpy as np
SRC='/opt/pyvenv/lib/python3.13/site-packages/pyannote/audio/sample/sample.wav'
if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'results'/'LOSSLESS_FLAC_REFERENCE.json'
    arr,sr=sf.read(SRC,dtype='int16')
    b=io.BytesIO(); sf.write(b,arr,sr,format='FLAC',subtype='PCM_16');b.seek(0)
    back,sr2=sf.read(b,dtype='int16')
    exact=(sr2==sr) and np.array_equal(arr,back)
    result={'original_wav_file_bytes':Path(SRC).stat().st_size,'original_pcm_bytes':int(arr.nbytes),
            'flac_bytes':int(b.getbuffer().nbytes),'sr_Hz':sr,'duration_s':len(arr)/sr,'channels':1,
            'PCM16_exact_roundtrip':bool(exact),'lossless_flac_ratio_vs_raw_pcm':float(arr.nbytes/b.getbuffer().nbytes),
            'input_sha256':hashlib.sha256(Path(SRC).read_bytes()).hexdigest(),
            'note':'FLAC is a waveform/PCM exact reconstruction baseline, the Mel feature AE cannot produce waveform or preserve its phase.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result))
