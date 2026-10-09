from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import run

def test_hamming74_has_unique_distance_three_codewords():
 c=run.hamming_codes();assert c.shape==(16,7)
 dist=(c[:,None,:]!=c[None,:,:]).sum(2);dist[torch.eye(16,dtype=torch.bool)]=99;assert int(dist.min())==3

def test_hamming74_corrects_each_single_bit_error():
 c=run.hamming_codes();noisy=c[:,None,:].expand(16,7,7).clone()
 for bit in range(7):noisy[:,bit,bit]^=1
 decoded=run.decode('hamming74',noisy.reshape(-1,7),c);assert torch.equal(decoded,torch.arange(16).repeat_interleave(7))

def test_repeat_three_majority_corrects_each_single_bit_error():
 c=run.codebook('repeat3',49401);x=c[:,None,:].expand(16,12,12).clone()
 for bit in range(12):x[:,bit,bit]^=1
 decoded=run.decode('repeat3',x.reshape(-1,12),c);assert torch.equal(decoded,torch.arange(16).repeat_interleave(12))
